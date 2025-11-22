#!/usr/bin/env python
"""
对比不同融合机制的效果
通过消融实验直观展示各个模块的作用
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

from models.alignn import ALIGNN, ALIGNNConfig
from data import get_train_val_loaders

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


class FusionComparator:
    """融合机制对比器"""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def extract_features_ablation(self, data_loader, max_samples=None):
        """
        提取不同阶段的特征（消融实验）

        Returns:
            features_dict: {
                'graph_base': 图基础特征（无任何融合）,
                'text_base': 文本基础特征（无任何融合）,
                'graph_middle': 应用中间融合后的图特征,
                'graph_fine': 应用细粒度注意力后的图特征,
                'text_fine': 应用细粒度注意力后的文本特征,
                'graph_cross': 应用全局注意力后的图特征,
                'text_cross': 应用全局注意力后的文本特征,
                'fused': 最终融合特征
            }
            targets: 目标值
            ids: 样本ID
        """
        print("🔄 提取不同阶段的特征（消融实验）...")

        # 保存原始配置
        original_middle = self.model.use_middle_fusion
        original_fine = self.model.use_fine_grained_attention
        original_cross = self.model.use_cross_modal_attention

        features_dict = {
            'graph_base': [],
            'text_base': [],
            'graph_middle': [],
            'graph_fine': [],
            'text_fine': [],
            'graph_cross': [],
            'text_cross': [],
            'fused': []
        }
        targets = []
        ids = []

        sample_count = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(data_loader, desc="提取特征")):
                if len(batch) == 3:
                    g, text, target = batch
                    lg = None
                elif len(batch) == 4:
                    g, lg, text, target = batch
                else:
                    raise ValueError(f"不支持的batch格式")

                g = g.to(self.device)
                if lg is not None:
                    lg = lg.to(self.device)

                # 处理text
                if isinstance(text, dict):
                    text = {k: v.to(self.device) for k, v in text.items()}
                elif isinstance(text, (list, tuple)):
                    # text是字符串列表，保持不动
                    pass
                elif torch.is_tensor(text):
                    text = text.to(self.device)

                batch_size = target.size(0)

                # 构建模型输入
                if lg is not None:
                    model_input = (g, lg, text)
                else:
                    model_input = (g, text)

                # ========== 1. 基础特征（关闭所有融合）==========
                self.model.use_middle_fusion = False
                self.model.use_fine_grained_attention = False
                self.model.use_cross_modal_attention = False

                output_base = self.model(model_input, return_features=True)
                features_dict['graph_base'].append(output_base['graph_features'].cpu().numpy())
                features_dict['text_base'].append(output_base['text_features'].cpu().numpy())

                # ========== 2. 中间融合特征 ==========
                if original_middle:
                    self.model.use_middle_fusion = True
                    self.model.use_fine_grained_attention = False
                    self.model.use_cross_modal_attention = False

                    output_middle = self.model(model_input, return_features=True)
                    features_dict['graph_middle'].append(output_middle['graph_features'].cpu().numpy())

                # ========== 3. 细粒度注意力特征 ==========
                if original_fine:
                    self.model.use_middle_fusion = original_middle  # 保留中间融合
                    self.model.use_fine_grained_attention = True
                    self.model.use_cross_modal_attention = False

                    output_fine = self.model(model_input, return_features=True)
                    features_dict['graph_fine'].append(output_fine['graph_features'].cpu().numpy())
                    features_dict['text_fine'].append(output_fine['text_features'].cpu().numpy())

                # ========== 4. 全局注意力特征 ==========
                if original_cross:
                    self.model.use_middle_fusion = original_middle
                    self.model.use_fine_grained_attention = original_fine
                    self.model.use_cross_modal_attention = True

                    output_cross = self.model(model_input, return_features=True)
                    features_dict['graph_cross'].append(output_cross['graph_features'].cpu().numpy())
                    features_dict['text_cross'].append(output_cross['text_features'].cpu().numpy())

                    # 融合特征
                    fused = np.concatenate([
                        output_cross['graph_features'].cpu().numpy(),
                        output_cross['text_features'].cpu().numpy()
                    ], axis=1)
                    features_dict['fused'].append(fused)

                targets.append(target.cpu().numpy())

                # 记录样本ID（如果有）
                if hasattr(g, 'ndata') and 'jid' in g.ndata:
                    batch_ids = [g.ndata['jid'][i] for i in range(batch_size)]
                    ids.extend(batch_ids)

                sample_count += batch_size
                if max_samples and sample_count >= max_samples:
                    break

        # 恢复原始配置
        self.model.use_middle_fusion = original_middle
        self.model.use_fine_grained_attention = original_fine
        self.model.use_cross_modal_attention = original_cross

        # 合并所有batch
        for key in features_dict:
            if features_dict[key]:
                features_dict[key] = np.vstack(features_dict[key])
            else:
                features_dict[key] = None

        targets = np.concatenate(targets)

        print(f"✅ 特征提取完成，共 {len(targets)} 个样本")
        for key, feat in features_dict.items():
            if feat is not None:
                print(f"   {key}: {feat.shape}")

        return features_dict, targets, ids


def visualize_feature_comparison(features_dict, targets, save_dir, method='tsne',
                                 is_classification=False):
    """
    可视化不同融合策略的特征空间
    """
    print(f"\n🎨 使用{method.upper()}进行特征对比可视化...")

    # 定义要对比的特征组合
    comparisons = [
        ('graph_base', 'Graph (No Fusion)'),
        ('text_base', 'Text (No Fusion)'),
        ('graph_middle', 'Graph (+ Middle Fusion)'),
        ('graph_fine', 'Graph (+ Fine-grained Attn)'),
        ('graph_cross', 'Graph (+ Cross-modal Attn)'),
        ('fused', 'Fused (All Mechanisms)')
    ]

    # 过滤掉None的特征
    comparisons = [(k, t) for k, t in comparisons if features_dict.get(k) is not None]

    n_plots = len(comparisons)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 6*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # 对每种特征进行降维和可视化
    for idx, (feat_key, title) in enumerate(comparisons):
        ax = axes[idx]
        features = features_dict[feat_key]

        print(f"  降维 {feat_key}...")

        # t-SNE降维
        if method.lower() == 'tsne':
            embedded = TSNE(n_components=2, random_state=42,
                          perplexity=min(30, len(features)-1)).fit_transform(features)
        else:
            # 可以扩展支持UMAP
            embedded = TSNE(n_components=2, random_state=42,
                          perplexity=min(30, len(features)-1)).fit_transform(features)

        # 绘图
        if is_classification:
            unique_classes = np.unique(targets)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))

            for cls_idx, cls in enumerate(unique_classes):
                mask = targets == cls
                ax.scatter(embedded[mask, 0], embedded[mask, 1],
                          c=[colors[cls_idx]], label=f'Class {int(cls)}',
                          alpha=0.6, s=30, edgecolors='k', linewidth=0.3)
            ax.legend(loc='best', fontsize=8)
        else:
            scatter = ax.scatter(embedded[:, 0], embedded[:, 1],
                               c=targets, cmap='viridis', alpha=0.6,
                               s=30, edgecolors='k', linewidth=0.3)
            plt.colorbar(scatter, ax=ax, label='Target Value')

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title(title, fontsize=12, weight='bold')
        ax.grid(True, alpha=0.2)

    # 隐藏多余的子图
    for idx in range(len(comparisons), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Feature Space Comparison: Impact of Fusion Mechanisms',
                fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'feature_comparison_{method}.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存对比图: {save_path}")
    plt.close()


def compute_quantitative_metrics(features_dict, targets, is_classification=False):
    """
    计算定量评估指标

    指标:
    - Silhouette Score: 聚类质量（越高越好，-1到1）
    - Davies-Bouldin Index: 聚类分离度（越低越好）
    - Intra-class similarity: 类内相似度（越高越好）
    - Inter-class similarity: 类间相似度（越低越好）
    """
    print("\n📊 计算定量评估指标...")

    metrics = {}

    for feat_key, features in features_dict.items():
        if features is None:
            continue

        print(f"  评估 {feat_key}...")

        feat_metrics = {}

        # 1. Silhouette Score (需要至少2个类别)
        if len(np.unique(targets)) > 1:
            try:
                sil_score = silhouette_score(features, targets)
                feat_metrics['silhouette'] = sil_score
            except:
                feat_metrics['silhouette'] = None

        # 2. Davies-Bouldin Index
        if len(np.unique(targets)) > 1:
            try:
                db_score = davies_bouldin_score(features, targets)
                feat_metrics['davies_bouldin'] = db_score
            except:
                feat_metrics['davies_bouldin'] = None

        # 3. 类内/类间相似度（对于分类任务）
        if is_classification or len(np.unique(targets)) < 50:
            unique_labels = np.unique(targets)

            # 计算余弦相似度矩阵
            sim_matrix = cosine_similarity(features)

            intra_sims = []
            inter_sims = []

            for label in unique_labels:
                mask = targets == label
                indices = np.where(mask)[0]

                if len(indices) > 1:
                    # 类内相似度
                    intra_sim = sim_matrix[np.ix_(indices, indices)]
                    # 排除对角线（自己和自己）
                    intra_sim = intra_sim[~np.eye(len(indices), dtype=bool)]
                    if len(intra_sim) > 0:
                        intra_sims.append(np.mean(intra_sim))

                # 类间相似度
                other_mask = ~mask
                other_indices = np.where(other_mask)[0]
                if len(other_indices) > 0:
                    inter_sim = sim_matrix[np.ix_(indices, other_indices)]
                    inter_sims.append(np.mean(inter_sim))

            feat_metrics['intra_class_sim'] = np.mean(intra_sims) if intra_sims else None
            feat_metrics['inter_class_sim'] = np.mean(inter_sims) if inter_sims else None

            # 分离度 = 类内相似度 - 类间相似度（越高越好）
            if feat_metrics['intra_class_sim'] and feat_metrics['inter_class_sim']:
                feat_metrics['separation'] = (feat_metrics['intra_class_sim'] -
                                             feat_metrics['inter_class_sim'])

        metrics[feat_key] = feat_metrics

    return metrics


def visualize_metrics_comparison(metrics, save_dir):
    """可视化指标对比"""
    print("\n📊 绘制指标对比图...")

    # 准备数据
    metric_names = ['silhouette', 'davies_bouldin', 'intra_class_sim',
                    'inter_class_sim', 'separation']
    metric_labels = ['Silhouette↑', 'Davies-Bouldin↓', 'Intra-class Sim↑',
                     'Inter-class Sim↓', 'Separation↑']

    feat_keys = list(metrics.keys())

    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (metric_name, metric_label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]

        values = []
        labels = []
        colors = []

        color_map = {
            'graph_base': '#A8DADC',
            'text_base': '#FFDAB9',
            'graph_middle': '#7FB3D5',
            'graph_fine': '#F4A261',
            'text_fine': '#FFCBA4',
            'graph_cross': '#E76F51',
            'text_cross': '#FFAC86',
            'fused': '#6A4C93'
        }

        for feat_key in feat_keys:
            if metric_name in metrics[feat_key] and metrics[feat_key][metric_name] is not None:
                values.append(metrics[feat_key][metric_name])
                labels.append(feat_key.replace('_', '\n'))
                colors.append(color_map.get(feat_key, '#CCCCCC'))

        if values:
            bars = ax.bar(range(len(values)), values, color=colors,
                         edgecolor='black', linewidth=1.5, alpha=0.8)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric_label, fontsize=11, weight='bold')
            ax.set_title(metric_label, fontsize=12, weight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')

            # 标注数值
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    # 隐藏最后一个空白子图
    axes[-1].axis('off')

    plt.suptitle('Quantitative Metrics Comparison\n↑ Higher is better | ↓ Lower is better',
                fontsize=14, weight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'metrics_comparison.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存指标对比图: {save_path}")
    plt.close()


def generate_comparison_report(metrics, save_dir):
    """生成对比报告"""
    print("\n📝 生成对比报告...")

    report_path = os.path.join(save_dir, 'comparison_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("融合机制对比实验报告\n")
        f.write("="*80 + "\n\n")

        f.write("## 实验配置\n\n")
        f.write("对比的融合策略:\n")
        f.write("  1. graph_base: 图基础特征（无融合）\n")
        f.write("  2. text_base: 文本基础特征（无融合）\n")
        f.write("  3. graph_middle: 图特征 + 中间融合\n")
        f.write("  4. graph_fine: 图特征 + 细粒度注意力\n")
        f.write("  5. graph_cross: 图特征 + 全局跨模态注意力\n")
        f.write("  6. fused: 完整融合（所有机制）\n\n")

        f.write("## 定量评估结果\n\n")
        f.write("指标说明:\n")
        f.write("  - Silhouette Score: 聚类质量（-1到1，越高越好）\n")
        f.write("  - Davies-Bouldin Index: 聚类分离度（越低越好）\n")
        f.write("  - Intra-class Similarity: 类内相似度（0到1，越高越好）\n")
        f.write("  - Inter-class Similarity: 类间相似度（0到1，越低越好）\n")
        f.write("  - Separation: 分离度 = 类内相似度 - 类间相似度（越高越好）\n\n")

        # 创建表格
        f.write("-" * 80 + "\n")
        f.write(f"{'Feature Type':<20} {'Silhouette':>12} {'Davies-B':>12} "
                f"{'Intra-Sim':>12} {'Inter-Sim':>12} {'Separation':>12}\n")
        f.write("-" * 80 + "\n")

        for feat_key, feat_metrics in metrics.items():
            sil = feat_metrics.get('silhouette', None)
            db = feat_metrics.get('davies_bouldin', None)
            intra = feat_metrics.get('intra_class_sim', None)
            inter = feat_metrics.get('inter_class_sim', None)
            sep = feat_metrics.get('separation', None)

            f.write(f"{feat_key:<20} "
                   f"{sil:>12.4f} " if sil else f"{feat_key:<20} {'N/A':>12} ")
            f.write(f"{db:>12.4f} " if db else f"{'N/A':>12} ")
            f.write(f"{intra:>12.4f} " if intra else f"{'N/A':>12} ")
            f.write(f"{inter:>12.4f} " if inter else f"{'N/A':>12} ")
            f.write(f"{sep:>12.4f}\n" if sep else f"{'N/A':>12}\n")

        f.write("-" * 80 + "\n\n")

        # 分析最佳配置
        f.write("## 分析结论\n\n")

        # 找出最佳配置
        if metrics:
            # Silhouette最高
            sil_scores = {k: v.get('silhouette') for k, v in metrics.items()
                         if v.get('silhouette') is not None}
            if sil_scores:
                best_sil = max(sil_scores.items(), key=lambda x: x[1])
                f.write(f"✅ Silhouette Score最高: {best_sil[0]} ({best_sil[1]:.4f})\n")

            # Davies-Bouldin最低
            db_scores = {k: v.get('davies_bouldin') for k, v in metrics.items()
                        if v.get('davies_bouldin') is not None}
            if db_scores:
                best_db = min(db_scores.items(), key=lambda x: x[1])
                f.write(f"✅ Davies-Bouldin最低: {best_db[0]} ({best_db[1]:.4f})\n")

            # Separation最高
            sep_scores = {k: v.get('separation') for k, v in metrics.items()
                         if v.get('separation') is not None}
            if sep_scores:
                best_sep = max(sep_scores.items(), key=lambda x: x[1])
                f.write(f"✅ Separation最高: {best_sep[0]} ({best_sep[1]:.4f})\n")

        f.write("\n")
        f.write("## 建议\n\n")
        f.write("1. 对于需要良好聚类质量的任务，建议使用完整融合配置\n")
        f.write("2. 如果计算资源有限，可以只使用关键的融合机制\n")
        f.write("3. 通过可视化图片可以更直观地观察特征空间的变化\n")

    print(f"✅ 报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='对比不同融合机制的效果')

    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['jarvis', 'mp', 'class'],
                       help='数据集类型')
    parser.add_argument('--property', type=str, required=True,
                       help='属性名称')
    parser.add_argument('--root_dir', type=str, default='./dataset',
                       help='数据集根目录')

    # 可选参数
    parser.add_argument('--save_dir', type=str, default='./fusion_comparison',
                       help='保存结果的目录')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu 或 cuda)')
    parser.add_argument('--max_samples', type=int, default=500,
                       help='最大样本数（用于加速）')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='使用哪个数据集split')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("🔬 融合机制对比实验")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}/{args.property}")
    print(f"Save dir: {args.save_dir}")
    print(f"Max samples: {args.max_samples}")
    print()

    # 1. 加载模型
    print("📥 加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_config = checkpoint.get('config', None)

    if model_config is None:
        print("❌ Checkpoint中未找到模型配置")
        return

    model = ALIGNN(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✅ 模型加载成功")

    is_classification = model_config.classification if hasattr(model_config, 'classification') else False
    print(f"任务类型: {'分类' if is_classification else '回归'}")

    # 检查模型支持的融合机制
    print(f"\n模型配置的融合机制:")
    print(f"  - 对比学习: {model_config.use_contrastive_loss if hasattr(model_config, 'use_contrastive_loss') else False}")
    print(f"  - 中间融合: {model_config.use_middle_fusion if hasattr(model_config, 'use_middle_fusion') else False}")
    print(f"  - 细粒度注意力: {model_config.use_fine_grained_attention if hasattr(model_config, 'use_fine_grained_attention') else False}")
    print(f"  - 全局注意力: {model_config.use_cross_modal_attention if hasattr(model_config, 'use_cross_modal_attention') else False}")

    # 2. 加载数据
    print("\n📊 加载数据...")
    from train_with_cross_modal_attention import load_dataset, get_dataset_paths

    dataset_mapping = {'jarvis': 'jarvis', 'mp': 'mp', 'class': 'class'}
    actual_dataset = dataset_mapping.get(args.dataset.lower(), args.dataset.lower())

    cif_dir, id_prop_file = get_dataset_paths(args.root_dir, actual_dataset, args.property)
    df = load_dataset(cif_dir, id_prop_file, actual_dataset, args.property)
    print(f"✅ 数据集大小: {len(df)}")

    # 创建数据加载器
    (train_loader, val_loader, test_loader, _) = get_train_val_loaders(
        dataset='user_data',
        dataset_array=df,
        target='target',
        batch_size=args.batch_size,
        atom_features=model_config.atom_features if hasattr(model_config, 'atom_features') else 'cgcnn',
        neighbor_strategy='k-nearest',
        line_graph=model_config.line_graph if hasattr(model_config, 'line_graph') else True,
        split_seed=42,
        workers=4,
        pin_memory=False,
        save_dataloader=False,
        filename='temp',
        id_tag='jid',
        use_canonize=True,
        cutoff=8.0,
        max_neighbors=12,
        output_dir=args.save_dir
    )

    # 选择数据集
    if args.split == 'train':
        data_loader = train_loader
    elif args.split == 'val':
        data_loader = val_loader
    else:
        data_loader = test_loader

    print(f"✅ 使用{args.split}集")

    # 3. 提取特征（消融实验）
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device = 'cpu'

    comparator = FusionComparator(model, device=device)
    features_dict, targets, ids = comparator.extract_features_ablation(
        data_loader, max_samples=args.max_samples
    )

    # 4. 可视化对比
    visualize_feature_comparison(features_dict, targets, args.save_dir,
                                method='tsne', is_classification=is_classification)

    # 5. 定量评估
    metrics = compute_quantitative_metrics(features_dict, targets,
                                          is_classification=is_classification)

    # 6. 可视化指标
    visualize_metrics_comparison(metrics, args.save_dir)

    # 7. 生成报告
    generate_comparison_report(metrics, args.save_dir)

    print("\n" + "="*80)
    print("✅ 对比实验完成！")
    print("="*80)
    print(f"\n生成的文件在: {args.save_dir}")
    print("  - feature_comparison_tsne.pdf/png: 特征空间对比可视化")
    print("  - metrics_comparison.pdf/png: 定量指标对比")
    print("  - comparison_report.txt: 详细对比报告")


if __name__ == '__main__':
    main()
