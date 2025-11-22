#!/usr/bin/env python
"""
从训练输出目录直接进行潜在空间可视化
无需手动指定data loader，自动从输出目录重建数据集
"""

import os
import sys
import argparse
import torch
import warnings
warnings.filterwarnings('ignore')

# 导入必要的模块
from data import get_train_val_loaders
from config import TrainingConfig
from models.alignn import ALIGNN, ALIGNNConfig
import visualize_latent_space as vis


def load_training_config(output_dir):
    """从输出目录加载训练配置"""
    # 尝试从history文件中获取配置信息
    import json

    # 检查是否有config.json
    config_file = os.path.join(output_dir, 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return config_dict

    # 如果没有，提示用户需要手动指定参数
    return None


def main():
    parser = argparse.ArgumentParser(description='从训练输出目录进行潜在空间可视化')

    # 必需参数
    parser.add_argument('--output_dir', type=str, required=True,
                        help='训练输出目录（包含best_val_model.pt等）')

    # 数据集参数
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['jarvis', 'mp', 'class'],
                        help='数据集类型')
    parser.add_argument('--property', type=str, required=True,
                        help='属性名称（如 formation_energy, syn等）')
    parser.add_argument('--root_dir', type=str, default='./dataset',
                        help='数据集根目录')

    # 可选参数
    parser.add_argument('--checkpoint', type=str, default='best_val_model.pt',
                        help='checkpoint文件名（在output_dir中）')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='使用哪个数据集split')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='保存图片的目录（默认为output_dir/latent_space_vis）')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备 (cpu 或 cuda)')
    parser.add_argument('--feature_types', nargs='+',
                        default=['graph', 'text', 'fused'],
                        choices=['graph', 'text', 'fused'],
                        help='要可视化的特征类型')
    parser.add_argument('--methods', nargs='+', default=['tsne'],
                        choices=['tsne', 'umap'],
                        help='降维方法（默认只用tsne，因为umap需要额外安装）')
    parser.add_argument('--dimensions', nargs='+', type=int, default=[2],
                        choices=[2, 3],
                        help='降维维度（默认只2D，3D可能较慢）')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数（用于大数据集，加速可视化）')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载worker数')

    args = parser.parse_args()

    # 设置保存目录
    if args.save_dir is None:
        args.save_dir = os.path.join(args.output_dir, 'latent_space_vis')

    # 检查checkpoint
    checkpoint_path = os.path.join(args.output_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        print(f"❌ 未找到checkpoint: {checkpoint_path}")
        return

    print("="*70)
    print("🎨 潜在空间可视化（从输出目录）")
    print("="*70)
    print(f"输出目录: {args.output_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"数据集: {args.dataset}/{args.property}")
    print(f"Split: {args.split}")
    print(f"保存目录: {args.save_dir}")
    print()

    # 1. 加载模型
    print("📥 加载模型...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_config = checkpoint.get('config', None)

    if model_config is None:
        print("❌ Checkpoint中未找到模型配置，无法重建模型")
        return

    model = ALIGNN(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✅ 模型加载成功")

    # 检测是否为分类任务
    is_classification = model_config.classification if hasattr(model_config, 'classification') else False
    print(f"任务类型: {'分类' if is_classification else '回归'}")

    # 2. 重建数据加载器
    print("\n📊 重建数据加载器...")

    try:
        # 导入数据加载函数
        from train_with_cross_modal_attention import load_dataset, get_dataset_paths

        # 根据数据集类型确定路径
        dataset_mapping = {
            'jarvis': 'jarvis',
            'mp': 'mp',
            'class': 'class'
        }

        actual_dataset = dataset_mapping.get(args.dataset.lower(), args.dataset.lower())

        # 获取数据集路径
        cif_dir, id_prop_file = get_dataset_paths(args.root_dir, actual_dataset, args.property)

        # 加载数据集
        df = load_dataset(cif_dir, id_prop_file, actual_dataset, args.property)
        print(f"✅ 加载数据集: {len(df)} 样本")

        # 如果设置了max_samples，进行采样
        if args.max_samples and len(df) > args.max_samples:
            print(f"⚠️  数据集过大，随机采样 {args.max_samples} 样本用于可视化")
            import random
            random.seed(42)
            df = random.sample(df, args.max_samples)

        # 直接创建数据加载器，不使用TrainingConfig（避免dataset限制）
        (train_loader, val_loader, test_loader,
         prepare_batch) = get_train_val_loaders(
            dataset='user_data',  # 使用user_data避免dataset限制
            dataset_array=df,
            target='target',
            n_train=None,
            n_val=None,
            n_test=None,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            batch_size=args.batch_size,
            atom_features=model_config.atom_features if hasattr(model_config, 'atom_features') else 'cgcnn',
            neighbor_strategy='k-nearest',
            line_graph=model_config.line_graph if hasattr(model_config, 'line_graph') else True,
            split_seed=42,
            workers=args.num_workers,
            pin_memory=False,
            save_dataloader=False,
            filename='temp_vis',
            id_tag='jid',
            use_canonize=True,
            cutoff=8.0,
            max_neighbors=12,
            output_dir=args.output_dir,
        )

        # 获取数据集对象
        train_data = train_loader.dataset
        val_data = val_loader.dataset
        test_data = test_loader.dataset

        # 选择要可视化的split
        if args.split == 'train':
            data_loader = train_loader
            print(f"✅ 使用训练集: {len(train_data)} 样本")
        elif args.split == 'val':
            data_loader = val_loader
            print(f"✅ 使用验证集: {len(val_data)} 样本")
        else:
            data_loader = test_loader
            print(f"✅ 使用测试集: {len(test_data)} 样本")

    except Exception as e:
        print(f"❌ 重建数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. 运行可视化
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device = 'cpu'

    vis.visualize_latent_space(
        checkpoint_path=checkpoint_path,
        data_loader=data_loader,
        save_dir=args.save_dir,
        device=device,
        feature_types=args.feature_types,
        methods=args.methods,
        dimensions=args.dimensions,
        is_classification=is_classification
    )


if __name__ == '__main__':
    main()
