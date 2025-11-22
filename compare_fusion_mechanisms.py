#!/usr/bin/env python
"""
对比不同融合机制的效果
通过消融实验直观展示各个模块的作用
版本2: 使用return_intermediate_features参数，避免动态修改模型架构
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
                'graph_base': 图基础特征（投影后，融合前）,
                'text_base': 文本基础特征（投影后，融合前）,
                'graph_cross': 应用全局注意力后的图特征（如果启用）,
                'text_cross': 应用全局注意力后的文本特征（如果启用）,
                'graph_final': 最终图特征,
                'text_final': 最终文本特征,
                'fused': 最终融合特征
            }
            targets: 目标值
            ids: 样本ID
        """
        print("🔄 提取不同阶段的特征（消融实验）...")

        # 检查模型配置
        has_middle = self.model.use_middle_fusion
        has_fine = self.model.use_fine_grained_attention
        has_cross = self.model.use_cross_modal_attention

        print(f"   模型配置: 中间融合={has_middle}, 细粒度注意力={has_fine}, 全局注意力={has_cross}")

        features_dict = {
            'graph_base': [],
            'text_base': [],
            'graph_cross': [],
            'text_cross': [],
            'graph_final': [],
            'text_final': [],
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
                    raise ValueError(f"不支持的batch格式: {len(batch)}个元素")

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

                # 提取中间特征（使用新的return_intermediate_features参数）
                output = self.model(model_input, return_intermediate_features=True)

                # 基础特征（融合前）
                features_dict['graph_base'].append(output['graph_base'].cpu().numpy())
                features_dict['text_base'].append(output['text_base'].cpu().numpy())

                # 最终特征
                features_dict['graph_final'].append(output['graph_features'].cpu().numpy())
                features_dict['text_final'].append(output['text_features'].cpu().numpy())

                # 全局注意力后的特征（如果启用）
                if has_cross and 'graph_cross' in output:
                    features_dict['graph_cross'].append(output['graph_cross'].cpu().numpy())
                    features_dict['text_cross'].append(output['text_cross'].cpu().numpy())

                # 融合特征
                fused = np.concatenate([
                    output['graph_features'].cpu().numpy(),
                    output['text_features'].cpu().numpy()
                ], axis=1)
                features_dict['fused'].append(fused)

                targets.append(target.cpu().numpy())

                # 记录样本ID（如果有）
                if hasattr(g, 'ndata') and 'jid' in g.ndata:
                    batch_ids = [g.ndata['jid'][i] for i in range(g.batch_size)]
                    ids.extend(batch_ids)

                sample_count += batch_size
                if max_samples and sample_count >= max_samples:
                    break

        # 转换为numpy数组
        for key in features_dict:
            if len(features_dict[key]) > 0:
                features_dict[key] = np.concatenate(features_dict[key], axis=0)
            else:
                features_dict[key] = None

        targets = np.concatenate(targets, axis=0)

        # 移除空特征
        features_dict = {k: v for k, v in features_dict.items() if v is not None}

        print(f"✅ 提取完成! 样本数: {len(targets)}, 特征类型: {list(features_dict.keys())}")

        return features_dict, targets, ids

    def visualize_tsne(self, features_dict, targets, save_dir):
        """使用t-SNE可视化不同阶段的特征"""
        print("\n📊 生成t-SNE可视化...")

        # 确定要可视化的特征
        feature_names = []
        feature_data = []

        if 'graph_base' in features_dict:
            feature_names.append('Graph Base')
            feature_data.append(features_dict['graph_base'])

        if 'text_base' in features_dict:
            feature_names.append('Text Base')
            feature_data.append(features_dict['text_base'])

        if 'graph_cross' in features_dict:
            feature_names.append('Graph + Cross-Modal')
            feature_data.append(features_dict['graph_cross'])

        if 'text_cross' in features_dict:
            feature_names.append('Text + Cross-Modal')
            feature_data.append(features_dict['text_cross'])

        if 'graph_final' in features_dict:
            feature_names.append('Graph Final')
            feature_data.append(features_dict['graph_final'])

        if 'text_final' in features_dict:
            feature_names.append('Text Final')
            feature_data.append(features_dict['text_final'])

        if 'fused' in features_dict:
            feature_names.append('Fused')
            feature_data.append(features_dict['fused'])

        n_features = len(feature_names)
        if n_features == 0:
            print("⚠️  没有可视化的特征!")
            return

        # 创建网格布局
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_features > 1 else [axes]

        # 对每个特征进行t-SNE
        for idx, (name, features) in enumerate(zip(feature_names, feature_data)):
            print(f"   处理 {name}...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = tsne.fit_transform(features)

            ax = axes[idx]
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                                c=targets, cmap='viridis', alpha=0.6, s=20)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            plt.colorbar(scatter, ax=ax, label='Target Value')

        # 隐藏多余的子图
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'tsne_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ t-SNE可视化已保存: {save_path}")
        plt.close()

    def compute_metrics(self, features_dict, targets, save_dir):
        """计算不同特征的质量指标"""
        print("\n📈 计算特征质量指标...")

        metrics_list = []

        for name, features in features_dict.items():
            if features is None or len(features) == 0:
                continue

            print(f"   分析 {name}...")

            # Silhouette Score (轮廓系数, 越大越好)
            try:
                sil_score = silhouette_score(features, targets)
            except:
                sil_score = np.nan

            # Davies-Bouldin Index (越小越好)
            try:
                db_score = davies_bouldin_score(features, targets)
            except:
                db_score = np.nan

            # Intra-class similarity (类内相似度, 越大越好)
            intra_sim = self._compute_intra_class_similarity(features, targets)

            # Inter-class separation (类间分离度, 越大越好)
            inter_sep = self._compute_inter_class_separation(features, targets)

            metrics_list.append({
                'Feature': name,
                'Silhouette Score': sil_score,
                'Davies-Bouldin Index': db_score,
                'Intra-class Similarity': intra_sim,
                'Inter-class Separation': inter_sep
            })

        # 创建DataFrame
        df = pd.DataFrame(metrics_list)
        save_path = os.path.join(save_dir, 'feature_metrics.csv')
        df.to_csv(save_path, index=False)
        print(f"\n✅ 指标已保存: {save_path}")
        print("\n" + df.to_string(index=False))

        # 可视化指标
        self._plot_metrics(df, save_dir)

        return df

    def _compute_intra_class_similarity(self, features, targets):
        """计算类内相似度"""
        unique_targets = np.unique(targets)
        if len(unique_targets) < 2:
            return 1.0

        sims = []
        for target in unique_targets[:10]:  # 只取前10个类别避免计算过慢
            mask = targets == target
            if np.sum(mask) < 2:
                continue
            class_features = features[mask]
            sim_matrix = cosine_similarity(class_features)
            # 取上三角（不包括对角线）
            upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            sims.append(np.mean(upper_tri))

        return np.mean(sims) if len(sims) > 0 else 0.0

    def _compute_inter_class_separation(self, features, targets):
        """计算类间分离度"""
        unique_targets = np.unique(targets)
        if len(unique_targets) < 2:
            return 0.0

        # 计算每个类别的中心
        centroids = []
        for target in unique_targets[:10]:  # 只取前10个类别
            mask = targets == target
            if np.sum(mask) == 0:
                continue
            centroids.append(np.mean(features[mask], axis=0))

        if len(centroids) < 2:
            return 0.0

        centroids = np.array(centroids)
        # 计算中心之间的平均距离
        distances = []
        for i in range(len(centroids)):
            for j in range(i+1, len(centroids)):
                dist = np.linalg.norm(centroids[i] - centroids[j])
                distances.append(dist)

        return np.mean(distances)

    def _plot_metrics(self, df, save_dir):
        """可视化指标对比"""
        metrics = ['Silhouette Score', 'Davies-Bouldin Index',
                   'Intra-class Similarity', 'Inter-class Separation']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = df[['Feature', metric]].dropna()

            if len(data) == 0:
                continue

            x = range(len(data))
            y = data[metric].values
            labels = data['Feature'].values

            bars = ax.bar(x, y, alpha=0.7, color=sns.color_palette("husl", len(data)))
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

            # 标注数值
            for i, v in enumerate(y):
                ax.text(i, v + 0.01*max(y), f'{v:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_path = os.path.join(save_dir, 'metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 指标对比图已保存: {save_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='对比不同融合机制的效果 (v2)')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='dft_3d',
                        choices=['dft_3d', 'dft_2d', 'megnet', 'cfid_3d', 'qm9_std_jctc'],
                        help='JARVIS数据集名称 (默认: dft_3d)')
    parser.add_argument('--property', type=str, default='formation_energy_peratom',
                        help='目标属性')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--max_samples', type=int, default=500, help='最大样本数（用于快速测试）')
    parser.add_argument('--save_dir', type=str, default='./fusion_comparison',
                        help='结果保存目录')
    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载模型
    print(f"🔄 加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("Checkpoint中没有找到config")

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")

    # 创建模型
    model = ALIGNN(config)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    print(f"   模型配置:")
    print(f"     - 中间融合: {model.use_middle_fusion}")
    print(f"     - 细粒度注意力: {model.use_fine_grained_attention}")
    print(f"     - 全局注意力: {model.use_cross_modal_attention}")

    # 加载数据
    print(f"\n🔄 加载数据集: {args.dataset} - {args.property}")
    train_loader, val_loader, test_loader = get_train_val_loaders(
        dataset=args.dataset,
        target=args.property,
        n_train=None,
        n_val=None,
        n_test=None,
        batch_size=args.batch_size,
        workers=0,
        output_dir=args.save_dir
    )

    print(f"   测试集样本数: {len(test_loader.dataset)}")

    # 创建对比器
    comparator = FusionComparator(model, device=device)

    # 提取特征
    features_dict, targets, ids = comparator.extract_features_ablation(
        test_loader, max_samples=args.max_samples
    )

    # 可视化
    comparator.visualize_tsne(features_dict, targets, args.save_dir)

    # 计算指标
    metrics_df = comparator.compute_metrics(features_dict, targets, args.save_dir)

    print(f"\n🎉 分析完成! 结果保存在: {args.save_dir}")


if __name__ == '__main__':
    main()
