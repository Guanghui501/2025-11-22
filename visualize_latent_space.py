#!/usr/bin/env python
"""
潜在空间可视化 (Latent Space Visualization)
使用 t-SNE 和 UMAP 将模型学习到的特征嵌入降维到 2D/3D 空间
展示融合表示的质量和不同材料的区分能力
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 降维算法
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP未安装，将仅使用t-SNE。安装方法: pip install umap-learn")

# 设置绘图风格
sns.set_style("white")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.titlesize'] = 16


def load_model_and_data(checkpoint_path, config_file=None):
    """加载模型和配置"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 从checkpoint中恢复模型配置
    model_config = checkpoint.get('config', None)
    if model_config is None:
        raise ValueError("Checkpoint中未找到模型配置信息")

    # 重新构建模型
    from models.alignn import ALIGNN
    model = ALIGNN(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    return model, model_config


def extract_features(model, data_loader, device='cpu', feature_types=['fused']):
    """
    从模型中提取中间特征

    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 设备
        feature_types: 要提取的特征类型列表
                      ['graph', 'text', 'fused'] 或其子集

    Returns:
        features_dict: {feature_type: features_array}
        targets: 目标值数组
    """
    model = model.to(device)
    model.eval()

    features_dict = {ft: [] for ft in feature_types}
    targets = []

    print(f"🔄 正在提取特征: {feature_types}")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="提取特征"):
            # 解包batch
            if len(batch) == 3:
                g, text, target = batch
            elif len(batch) == 4:
                g, lg, text, target = batch
            else:
                raise ValueError(f"不支持的batch格式: {len(batch)}个元素")

            g = g.to(device)

            # 处理text输入（可能是dict, tensor, 或list）
            if isinstance(text, dict):
                text = {k: v.to(device) for k, v in text.items()}
            elif isinstance(text, (list, tuple)):
                # 如果是list/tuple，每个元素可能是tensor
                text_processed = []
                for item in text:
                    if isinstance(item, dict):
                        text_processed.append({k: v.to(device) for k, v in item.items()})
                    elif torch.is_tensor(item):
                        text_processed.append(item.to(device))
                    else:
                        text_processed.append(item)
                text = text_processed
            elif torch.is_tensor(text):
                text = text.to(device)

            # Pack inputs for ALIGNN model
            # ALIGNN expects: (g, lg, text) or just g depending on batch format
            if len(batch) == 4:
                # Batch format: (g, lg, text, target)
                # Model expects: forward((g, lg, text), return_features=True)
                model_input = (g, batch[1].to(device), text)
            else:
                # Batch format: (g, text, target)
                # Model expects: forward((g, text), return_features=True)
                model_input = (g, text)

            # Forward pass with return_features=True
            output = model(model_input, return_features=True)

            if isinstance(output, dict):
                # 提取不同类型的特征
                if 'graph' in feature_types and 'graph_features' in output:
                    features_dict['graph'].append(output['graph_features'].cpu().numpy())

                if 'text' in feature_types and 'text_features' in output:
                    features_dict['text'].append(output['text_features'].cpu().numpy())

                if 'fused' in feature_types:
                    # 融合特征是graph和text特征的组合
                    if 'graph_features' in output and 'text_features' in output:
                        graph_feat = output['graph_features'].cpu().numpy()
                        text_feat = output['text_features'].cpu().numpy()
                        fused_feat = np.concatenate([graph_feat, text_feat], axis=1)
                        features_dict['fused'].append(fused_feat)

            targets.append(target.cpu().numpy())

    # 合并所有batch
    for ft in feature_types:
        if features_dict[ft]:
            features_dict[ft] = np.vstack(features_dict[ft])
        else:
            print(f"⚠️  未能提取 {ft} 特征")
            features_dict[ft] = None

    targets = np.concatenate(targets)

    print(f"✅ 特征提取完成")
    for ft, feat in features_dict.items():
        if feat is not None:
            print(f"   {ft}: {feat.shape}")
    print(f"   targets: {targets.shape}")

    return features_dict, targets


def apply_dimensionality_reduction(features, method='tsne', n_components=2, **kwargs):
    """
    应用降维算法

    Args:
        features: 特征矩阵 [n_samples, n_features]
        method: 'tsne' 或 'umap'
        n_components: 降维到的维度 (2 或 3)
        **kwargs: 传递给降维算法的其他参数

    Returns:
        embedded: 降维后的特征 [n_samples, n_components]
    """
    print(f"🔄 应用{method.upper()}降维到{n_components}D...")

    if method.lower() == 'tsne':
        default_params = {
            'n_components': n_components,
            'perplexity': min(30, len(features) - 1),
            'max_iter': 1000,  # 新版sklearn使用max_iter而不是n_iter
            'random_state': 42,
            'verbose': 0
        }
        default_params.update(kwargs)
        reducer = TSNE(**default_params)

    elif method.lower() == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP未安装，请使用: pip install umap-learn")

        default_params = {
            'n_components': n_components,
            'n_neighbors': min(15, len(features) - 1),
            'min_dist': 0.1,
            'random_state': 42,
            'verbose': False
        }
        default_params.update(kwargs)
        reducer = umap.UMAP(**default_params)
    else:
        raise ValueError(f"不支持的降维方法: {method}")

    embedded = reducer.fit_transform(features)
    print(f"✅ 降维完成: {embedded.shape}")

    return embedded


def plot_2d_embedding(embedded, targets, title, save_path, is_classification=False):
    """绘制2D嵌入空间"""
    fig, ax = plt.subplots(figsize=(10, 8))

    if is_classification:
        # 分类任务：按类别着色
        unique_classes = np.unique(targets)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

        for idx, cls in enumerate(unique_classes):
            mask = targets == cls
            ax.scatter(embedded[mask, 0], embedded[mask, 1],
                      c=[colors[idx]], label=f'Class {int(cls)}',
                      alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best')
    else:
        # 回归任务：按值着色
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                           c=targets, cmap='viridis', alpha=0.7,
                           s=50, edgecolors='k', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Target Value', rotation=270, labelpad=20)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
   # ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存2D图: {save_path}")
    plt.close()


def plot_3d_embedding(embedded, targets, title, save_path, is_classification=False):
    """绘制3D嵌入空间"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    if is_classification:
        # 分类任务：按类别着色
        unique_classes = np.unique(targets)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

        for idx, cls in enumerate(unique_classes):
            mask = targets == cls
            ax.scatter(embedded[mask, 0], embedded[mask, 1], embedded[mask, 2],
                      c=[colors[idx]], label=f'Class {int(cls)}',
                      alpha=0.7, s=50, edgecolors='k', linewidth=0.5)
        ax.legend(loc='best')
    else:
        # 回归任务：按值着色
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
                           c=targets, cmap='viridis', alpha=0.7,
                           s=50, edgecolors='k', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Target Value', rotation=270, labelpad=20)

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_zlabel('Dimension 3')
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存3D图: {save_path}")
    plt.close()


def plot_comparison(embeddings_dict, targets, method, save_path, is_classification=False):
    """
    并排对比不同特征类型的嵌入空间

    Args:
        embeddings_dict: {feature_type: embedded_2d}
        targets: 目标值
        method: 降维方法名称
        save_path: 保存路径
        is_classification: 是否为分类任务
    """
    n_plots = len(embeddings_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))

    if n_plots == 1:
        axes = [axes]

    titles = {
        'graph': 'Graph Features Only',
        'text': 'Text Features Only',
        'fused': 'Fused Features (Graph + Text)'
    }

    # 计算所有嵌入的全局坐标范围，用于统一横纵坐标
    all_embeddings = np.vstack(list(embeddings_dict.values()))
    x_min, x_max = all_embeddings[:, 0].min(), all_embeddings[:, 0].max()
    y_min, y_max = all_embeddings[:, 1].min(), all_embeddings[:, 1].max()

    # 添加10%的padding
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    x_lim = [x_min - x_margin, x_max + x_margin]
    y_lim = [y_min - y_margin, y_max + y_margin]

    for idx, (feat_type, embedded) in enumerate(embeddings_dict.items()):
        ax = axes[idx]

        if is_classification:
            # 分类任务
            unique_classes = np.unique(targets)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

            for cls_idx, cls in enumerate(unique_classes):
                mask = targets == cls
                ax.scatter(embedded[mask, 0], embedded[mask, 1],
                          c=[colors[cls_idx]], label=f'Class {int(cls)}',
                          alpha=0.7, s=40, edgecolors='k', linewidth=0.5)
            ax.legend(loc='best', fontsize=9)
        else:
            # 回归任务
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                               c=targets, cmap='viridis', alpha=0.7,
                               s=40, edgecolors='k', linewidth=0.5)
            if idx == n_plots - 1:  # 只在最后一个子图添加colorbar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Target Value', rotation=270, labelpad=15)

        # 统一设置横纵坐标范围
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(titles.get(feat_type, feat_type))
       # ax.grid(True, alpha=0.3)

    plt.suptitle(f'Latent Space Comparison ({method.upper()})', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存对比图: {save_path}")
    plt.close()


def visualize_latent_space(checkpoint_path, data_loader, save_dir, device='cpu',
                           feature_types=['graph', 'text', 'fused'],
                           methods=['tsne', 'umap'], dimensions=[2, 3],
                           is_classification=False):
    """
    完整的潜在空间可视化流程

    Args:
        checkpoint_path: 模型checkpoint路径
        data_loader: 数据加载器
        save_dir: 保存目录
        device: 设备
        feature_types: 要可视化的特征类型
        methods: 降维方法
        dimensions: 降维维度
        is_classification: 是否为分类任务
    """
    os.makedirs(save_dir, exist_ok=True)

    print("="*70)
    print("🎨 潜在空间可视化")
    print("="*70)

    # 1. 加载模型
    print("📥 加载模型...")
    model, model_config = load_model_and_data(checkpoint_path)
    model = model.to(device)

    # 2. 提取特征
    features_dict, targets = extract_features(model, data_loader, device, feature_types)

    # 过滤掉None的特征
    features_dict = {k: v for k, v in features_dict.items() if v is not None}

    if not features_dict:
        print("❌ 未能提取任何特征，退出")
        return

    # 3. 对每种特征类型和降维方法进行可视化
    embeddings_2d = {}

    for method in methods:
        if method == 'umap' and not UMAP_AVAILABLE:
            print(f"⚠️  跳过{method.upper()}（未安装）")
            continue

        print(f"\n{'='*70}")
        print(f"📊 使用 {method.upper()} 进行降维")
        print(f"{'='*70}")

        embeddings_2d_method = {}

        for feat_type, features in features_dict.items():
            print(f"\n--- 处理 {feat_type} 特征 ---")

            # 2D降维
            if 2 in dimensions:
                embedded_2d = apply_dimensionality_reduction(
                    features, method=method, n_components=2
                )
                embeddings_2d_method[feat_type] = embedded_2d

                # 单独绘制
                save_path = os.path.join(save_dir,
                    f'latent_space_{feat_type}_{method}_2d.pdf')
                title = f'{feat_type.capitalize()} Features - {method.upper()} 2D'
                plot_2d_embedding(embedded_2d, targets, title, save_path, is_classification)

            # 3D降维
            if 3 in dimensions:
                embedded_3d = apply_dimensionality_reduction(
                    features, method=method, n_components=3
                )

                save_path = os.path.join(save_dir,
                    f'latent_space_{feat_type}_{method}_3d.pdf')
                title = f'{feat_type.capitalize()} Features - {method.upper()} 3D'
                plot_3d_embedding(embedded_3d, targets, title, save_path, is_classification)

        # 绘制对比图（仅2D）
        if embeddings_2d_method and 2 in dimensions:
            comparison_path = os.path.join(save_dir,
                f'latent_space_comparison_{method}_2d.pdf')
            plot_comparison(embeddings_2d_method, targets, method,
                          comparison_path, is_classification)

    print("\n" + "="*70)
    print("✅ 潜在空间可视化完成！")
    print("="*70)
    print(f"所有图片已保存到: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='潜在空间可视化')

    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型checkpoint路径（如 best_val_model.pt）')
    parser.add_argument('--data_loader_path', type=str, required=True,
                        help='保存的data loader路径（如 data_loader_test.pt）')

    # 可选参数
    parser.add_argument('--save_dir', type=str, default='./latent_space_vis',
                        help='保存图片的目录')
    parser.add_argument('--device', type=str, default='cpu',
                        help='设备 (cpu 或 cuda)')
    parser.add_argument('--feature_types', nargs='+',
                        default=['graph', 'text', 'fused'],
                        choices=['graph', 'text', 'fused'],
                        help='要可视化的特征类型')
    parser.add_argument('--methods', nargs='+', default=['tsne', 'umap'],
                        choices=['tsne', 'umap'],
                        help='降维方法')
    parser.add_argument('--dimensions', nargs='+', type=int, default=[2, 3],
                        choices=[2, 3],
                        help='降维维度')
    parser.add_argument('--classification', action='store_true',
                        help='是否为分类任务')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小（如果需要重新创建data loader）')

    args = parser.parse_args()

    # 加载data loader
    print(f"📥 加载数据...")
    if os.path.exists(args.data_loader_path):
        data_loader = torch.load(args.data_loader_path, weights_only=False)
        print(f"✅ 从文件加载data loader: {args.data_loader_path}")
    else:
        print(f"❌ 未找到data loader文件: {args.data_loader_path}")
        print("提示: 训练时设置 --save_dataloader 参数来保存data loader")
        exit(1)

    # 检查设备
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device = 'cpu'

    # 运行可视化
    visualize_latent_space(
        checkpoint_path=args.checkpoint,
        data_loader=data_loader,
        save_dir=args.save_dir,
        device=device,
        feature_types=args.feature_types,
        methods=args.methods,
        dimensions=args.dimensions,
        is_classification=args.classification
    )
