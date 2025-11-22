#!/usr/bin/env python
"""
对比全模态和无中期融合模型的注意力权重

分析中期融合如何改善节点-文本对齐
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
import pandas as pd
from tqdm import tqdm
import argparse

# 添加路径
import sys
sys.path.insert(0, os.path.dirname(__file__))

from models.alignn import ALIGNN
from data import get_train_val_loaders


class AttentionComparator:
    """对比两个模型的注意力权重"""

    def __init__(self, checkpoint_full, checkpoint_no_middle, root_dir, device='cuda'):
        """
        Args:
            checkpoint_full: 全模态模型（中期+细粒度+全局）
            checkpoint_no_middle: 无中期融合模型（细粒度+全局）
            root_dir: 数据目录
        """
        self.device = device
        self.root_dir = root_dir

        # 加载模型
        print(f"加载模型...")
        print(f"  全模态: {checkpoint_full}")
        self.model_full = self._load_model(checkpoint_full)

        print(f"  无中期: {checkpoint_no_middle}")
        self.model_no_middle = self._load_model(checkpoint_no_middle)

        print("✅ 模型加载完成\n")

    def _load_model(self, checkpoint_path):
        """加载模型"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 打印 checkpoint 的键，方便调试
        print(f"    Checkpoint keys: {list(checkpoint.keys())}")

        # 获取配置
        if 'config' in checkpoint:
            config = checkpoint['config']
        elif 'model_config' in checkpoint:
            config = checkpoint['model_config']
        else:
            raise KeyError(f"Cannot find config in checkpoint. Available keys: {list(checkpoint.keys())}")

        # 重建模型
        model = ALIGNN(config)

        # 尝试多种可能的状态字典键名
        state_dict = None
        possible_keys = ['model_state_dict', 'state_dict', 'model', 'model_state']

        for key in possible_keys:
            if key in checkpoint:
                state_dict = checkpoint[key]
                print(f"    Found state dict with key: '{key}'")
                break

        if state_dict is None:
            # 如果都没找到，尝试直接使用 checkpoint（可能整个文件就是 state_dict）
            if all(isinstance(k, str) and not k.startswith('_') for k in checkpoint.keys()):
                state_dict = checkpoint
                print(f"    Using entire checkpoint as state dict")
            else:
                raise KeyError(f"Cannot find model state dict. Available keys: {list(checkpoint.keys())}")

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def extract_attention_weights(self, dataloader, num_samples=100):
        """
        从两个模型提取注意力权重

        Returns:
            dict: {
                'full_model': {
                    'fine_grained': [...],  # 细粒度注意力权重
                    'cross_modal': [...]     # 全局注意力权重
                },
                'no_middle_model': {...}
            }
        """
        results = {
            'full_model': {
                'fine_grained': [],
                'cross_modal': [],
                'sample_ids': []
            },
            'no_middle_model': {
                'fine_grained': [],
                'cross_modal': [],
                'sample_ids': []
            }
        }

        print(f"提取注意力权重（前 {num_samples} 个样本）...")

        count = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Processing"):
                if count >= num_samples:
                    break

                g, lg, text, target = batch
                g = g.to(self.device)
                if lg is not None:
                    lg = lg.to(self.device)

                batch_size = target.size(0)

                # 提取全模态模型的注意力
                model_input = (g, lg, text) if lg is not None else (g, text)
                output_full = self.model_full(model_input, return_attention=True)

                # 提取无中期模型的注意力
                output_no_middle = self.model_no_middle(model_input, return_attention=True)

                # 保存注意力权重
                # 细粒度注意力 (fine_grained_attention_weights 是一个字典)
                if 'fine_grained_attention_weights' in output_full:
                    # 使用 atom_to_text 注意力: [batch, heads, num_atoms, seq_len]
                    # 这显示每个原子关注哪些文本token
                    attn_full = output_full['fine_grained_attention_weights']['atom_to_text']
                    attn_no_middle = output_no_middle['fine_grained_attention_weights']['atom_to_text']

                    # 对多头注意力取平均: [batch, heads, num_atoms, seq_len] -> [batch, num_atoms, seq_len]
                    attn_full = attn_full.mean(dim=1)  # 平均所有注意力头
                    attn_no_middle = attn_no_middle.mean(dim=1)

                    results['full_model']['fine_grained'].append(
                        attn_full.cpu().numpy()
                    )
                    results['no_middle_model']['fine_grained'].append(
                        attn_no_middle.cpu().numpy()
                    )

                # 全局注意力 (cross-modal attention，也是一个字典)
                if 'attention_weights' in output_full:
                    # 使用 graph_to_text 注意力: [batch, heads, 1, 1]
                    attn_full = output_full['attention_weights']['graph_to_text']
                    attn_no_middle = output_no_middle['attention_weights']['graph_to_text']

                    # 对多头注意力取平均: [batch, heads, 1, 1] -> [batch, 1, 1]
                    attn_full = attn_full.mean(dim=1)
                    attn_no_middle = attn_no_middle.mean(dim=1)

                    results['full_model']['cross_modal'].append(
                        attn_full.cpu().numpy()
                    )
                    results['no_middle_model']['cross_modal'].append(
                        attn_no_middle.cpu().numpy()
                    )

                # 记录样本ID（只要有任何注意力权重）
                if 'fine_grained_attention_weights' in output_full or 'attention_weights' in output_full:
                    results['full_model']['sample_ids'].extend(range(count, count + batch_size))
                    results['no_middle_model']['sample_ids'].extend(range(count, count + batch_size))

                count += batch_size

        print(f"✅ 提取完成：{count} 个样本")
        print(f"  细粒度注意力: {len(results['full_model']['fine_grained'])} 批次")
        print(f"  全局注意力: {len(results['full_model']['cross_modal'])} 批次\n")

        return results

    def compute_attention_statistics(self, attention_weights):
        """
        计算注意力权重的统计指标

        Returns:
            dict: 各种统计指标
        """
        stats = {
            'entropy': [],           # 熵（分布集中度）
            'max_weight': [],        # 最大权重
            'effective_tokens': [],  # 有效token数（权重>0.1）
            'gini': []              # 基尼系数（不平等度）
        }

        for attn in attention_weights:
            # attn shape: [batch, num_atoms, num_tokens]
            for sample_attn in attn:
                # 对每个原子的注意力分布
                for atom_attn in sample_attn:
                    # 归一化
                    atom_attn = atom_attn / (atom_attn.sum() + 1e-8)

                    # 熵（越低越集中）
                    stats['entropy'].append(entropy(atom_attn + 1e-8))

                    # 最大权重
                    stats['max_weight'].append(atom_attn.max())

                    # 有效token数
                    stats['effective_tokens'].append((atom_attn > 0.1).sum())

                    # 基尼系数
                    sorted_attn = np.sort(atom_attn)
                    n = len(sorted_attn)
                    gini = (2 * np.sum((np.arange(n) + 1) * sorted_attn)) / (n * np.sum(sorted_attn)) - (n + 1) / n
                    stats['gini'].append(gini)

        # 计算平均值
        return {k: np.mean(v) for k, v in stats.items()}

    def visualize_attention_comparison(self, results, save_dir):
        """可视化注意力对比"""
        os.makedirs(save_dir, exist_ok=True)

        print("生成可视化...")

        # 1. 细粒度注意力统计对比
        if not results['full_model']['fine_grained']:
            print("⚠️  警告：未找到细粒度注意力权重，无法生成可视化")
            print("   请检查模型是否启用了 use_fine_grained_attention")
            return

        if results['full_model']['fine_grained']:
            stats_full = self.compute_attention_statistics(
                results['full_model']['fine_grained']
            )
            stats_no_middle = self.compute_attention_statistics(
                results['no_middle_model']['fine_grained']
            )

            # 绘制对比图
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('细粒度注意力权重对比：全模态 vs 无中期融合', fontsize=14, weight='bold')

            metrics = ['entropy', 'max_weight', 'effective_tokens', 'gini']
            titles = [
                '注意力熵（越低越集中）',
                '最大注意力权重（越高越明确）',
                '有效Token数（越少越选择性）',
                '基尼系数（越高越不平等）'
            ]

            for idx, (metric, title) in enumerate(zip(metrics, titles)):
                ax = axes[idx // 2, idx % 2]

                values = [stats_full[metric], stats_no_middle[metric]]
                labels = ['全模态\n(有中期融合)', '无中期融合']
                colors = ['#2ecc71', '#e74c3c']

                bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
                ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
                ax.set_title(title, fontsize=12, weight='bold')
                ax.grid(axis='y', alpha=0.3)

                # 添加数值标签
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.4f}',
                           ha='center', va='bottom', fontsize=10, weight='bold')

                # 添加改善百分比
                if stats_full[metric] != 0:
                    improvement = (stats_full[metric] - stats_no_middle[metric]) / abs(stats_no_middle[metric]) * 100
                    ax.text(0.5, 0.95, f'改善: {improvement:+.1f}%',
                           transform=ax.transAxes,
                           ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           fontsize=9)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/attention_statistics_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  ✅ 保存: attention_statistics_comparison.png")

            # 2. 绘制示例注意力热图
            self._plot_attention_heatmaps(results, save_dir)

            # 3. 保存统计数据
            comparison_df = pd.DataFrame({
                'Metric': metrics,
                'Full Model (with Middle Fusion)': [stats_full[m] for m in metrics],
                'No Middle Fusion': [stats_no_middle[m] for m in metrics],
                'Improvement (%)': [
                    (stats_full[m] - stats_no_middle[m]) / abs(stats_no_middle[m]) * 100
                    for m in metrics
                ]
            })
            comparison_df.to_csv(f'{save_dir}/attention_statistics.csv', index=False)
            print(f"  ✅ 保存: attention_statistics.csv")

            # 打印结果
            print("\n" + "="*80)
            print("📊 细粒度注意力权重统计对比")
            print("="*80)
            print(comparison_df.to_string(index=False))
            print("="*80 + "\n")

    def _plot_attention_heatmaps(self, results, save_dir, num_examples=3):
        """绘制注意力热图示例"""

        fine_grained_full = results['full_model']['fine_grained']
        fine_grained_no_middle = results['no_middle_model']['fine_grained']

        if not fine_grained_full:
            return

        # 选择几个示例
        for idx in range(min(num_examples, len(fine_grained_full))):
            attn_full = fine_grained_full[idx][0]  # 第一个样本
            attn_no_middle = fine_grained_no_middle[idx][0]

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f'示例 {idx+1}: 节点-Token 注意力热图对比', fontsize=14, weight='bold')

            # 全模态
            sns.heatmap(attn_full, cmap='YlOrRd', ax=axes[0],
                       cbar_kws={'label': 'Attention Weight'})
            axes[0].set_title('全模态（有中期融合）', fontsize=12)
            axes[0].set_xlabel('Text Tokens', fontsize=11)
            axes[0].set_ylabel('Graph Nodes', fontsize=11)

            # 无中期
            sns.heatmap(attn_no_middle, cmap='YlOrRd', ax=axes[1],
                       cbar_kws={'label': 'Attention Weight'})
            axes[1].set_title('无中期融合', fontsize=12)
            axes[1].set_xlabel('Text Tokens', fontsize=11)
            axes[1].set_ylabel('Graph Nodes', fontsize=11)

            plt.tight_layout()
            plt.savefig(f'{save_dir}/attention_heatmap_example_{idx+1}.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"  ✅ 保存: {num_examples} 个注意力热图示例")


def main():
    parser = argparse.ArgumentParser(description='对比注意力权重')
    parser.add_argument('--checkpoint_full', type=str, required=True,
                       help='全模态模型checkpoint')
    parser.add_argument('--checkpoint_no_middle', type=str, required=True,
                       help='无中期融合模型checkpoint')
    parser.add_argument('--root_dir', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--dataset', type=str, default='dft_3d',
                       help='数据集类型（如 dft_3d, mp 等）')
    parser.add_argument('--property', type=str, default='mbj_bandgap',
                       help='属性名称（如 mbj_bandgap, formation_energy 等）')
    parser.add_argument('--save_dir', type=str, default='./attention_comparison',
                       help='保存目录')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='分析的样本数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批大小')

    args = parser.parse_args()

    # 创建对比器
    comparator = AttentionComparator(
        args.checkpoint_full,
        args.checkpoint_no_middle,
        args.root_dir
    )

    # 加载数据集（使用与 compare_fusion_mechanisms.py 相同的方法）
    print(f"\n🔄 加载数据集: {args.dataset} - {args.property}")
    try:
        from train_with_cross_modal_attention import load_dataset, get_dataset_paths

        # 获取数据集路径
        cif_dir, id_prop_file = get_dataset_paths(args.root_dir, args.dataset, args.property)
        print(f"  数据文件: {id_prop_file}")
        print(f"  结构目录: {cif_dir}")

        # 加载数据集
        df = load_dataset(cif_dir, id_prop_file, args.dataset, args.property)
        print(f"✅  加载数据集: {len(df)} 样本")

        # 如果设置了num_samples，且小于数据集大小，进行采样
        if args.num_samples and len(df) > args.num_samples * 2:
            print(f"⚠️  数据集较大，为加快速度随机采样 {args.num_samples * 2} 样本")
            import random
            random.seed(42)
            df = random.sample(df, args.num_samples * 2)

        # 创建数据加载器（使用user_data避免dataset限制）
        train_loader, val_loader, test_loader, _ = get_train_val_loaders(
            dataset='user_data',
            dataset_array=df,
            target='target',
            n_train=None,
            n_val=None,
            n_test=None,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            batch_size=args.batch_size,
            atom_features='cgcnn',
            neighbor_strategy='k-nearest',
            line_graph=True,
            split_seed=42,
            workers=0,
            pin_memory=False,
            save_dataloader=False,
            filename='temp_attention_comparison',
            id_tag='jid',
            use_canonize=True,
            cutoff=8.0,
            max_neighbors=12,
            output_dir=args.save_dir
        )

        print(f"   测试集样本数: {len(test_loader.dataset)}")

    except Exception as e:
        print(f"❌  加载数据集失败: {e}")
        print("请确保:")
        print(f"  1. 数据集路径正确: {args.root_dir}")
        print(f"  2. 数据集类型正确: {args.dataset}")
        print(f"  3. 属性名称正确: {args.property}")
        raise

    # 提取注意力权重
    results = comparator.extract_attention_weights(test_loader, args.num_samples)

    # 可视化对比
    comparator.visualize_attention_comparison(results, args.save_dir)

    print(f"\n✅ 分析完成！结果保存在: {args.save_dir}")


if __name__ == '__main__':
    main()
