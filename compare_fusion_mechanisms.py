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
            'graph_middle': [],
            'graph_fine': [],
            'text_fine': [],
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

                # 中间融合后的特征（如果启用）
                if has_middle and 'graph_middle' in output:
                    features_dict['graph_middle'].append(output['graph_middle'].cpu().numpy())

                # 细粒度注意力后的特征（如果启用）
                if has_fine and 'graph_fine' in output:
                    features_dict['graph_fine'].append(output['graph_fine'].cpu().numpy())
                    features_dict['text_fine'].append(output['text_fine'].cpu().numpy())

                # 全局注意力后的特征（如果启用）
                if has_cross and 'graph_cross' in output:
                    features_dict['graph_cross'].append(output['graph_cross'].cpu().numpy())
                    features_dict['text_cross'].append(output['text_cross'].cpu().numpy())

                # 最终特征
                features_dict['graph_final'].append(output['graph_features'].cpu().numpy())
                features_dict['text_final'].append(output['text_features'].cpu().numpy())

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

        if 'graph_middle' in features_dict:
            feature_names.append('Graph + Middle Fusion')
            feature_data.append(features_dict['graph_middle'])

        if 'graph_fine' in features_dict:
            feature_names.append('Graph + Fine-grained Attn')
            feature_data.append(features_dict['graph_fine'])

        if 'text_fine' in features_dict:
            feature_names.append('Text + Fine-grained Attn')
            feature_data.append(features_dict['text_fine'])

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

        # 先对所有特征进行t-SNE，收集所有坐标用于统一坐标轴范围
        print("   第一步: 计算所有t-SNE嵌入...")
        all_features_2d = []
        for name, features in zip(feature_names, feature_data):
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            features_2d = tsne.fit_transform(features)
            all_features_2d.append(features_2d)

        # 计算全局坐标范围
        all_coords = np.vstack(all_features_2d)
        x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
        y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()

        # 添加边距
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        x_lim = [x_min - x_margin, x_max + x_margin]
        y_lim = [y_min - y_margin, y_max + y_margin]

        print(f"   全局坐标范围: x=[{x_lim[0]:.1f}, {x_lim[1]:.1f}], y=[{y_lim[0]:.1f}, {y_lim[1]:.1f}]")

        # 创建网格布局
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_features > 1 else [axes]

        # 第二步: 绘制每个特征
        print("   第二步: 绘制可视化...")
        for idx, (name, features_2d) in enumerate(zip(feature_names, all_features_2d)):
            ax = axes[idx]
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                                c=targets, cmap='viridis', alpha=0.6, s=20)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')

            # 设置统一的坐标范围
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)

            # 设置白色网格线
            ax.grid(True, color='white', linewidth=0.8, alpha=0.7)

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
    parser.add_argument('--dataset', type=str, required=True,
                        help='数据集类型 (jarvis/mp/class等)')
    parser.add_argument('--property', type=str, required=True,
                        help='目标属性 (如 formation_energy_peratom, bandgap等)')
    parser.add_argument('--root_dir', type=str, default='./dataset',
                        help='数据集根目录')
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

    # 加载数据集（支持本地数据）
    print(f"\n🔄 加载数据集: {args.dataset} - {args.property}")
    try:
        from train_with_cross_modal_attention import load_dataset, get_dataset_paths

        # 获取数据集路径
        cif_dir, id_prop_file = get_dataset_paths(args.root_dir, args.dataset, args.property)

        # 加载数据集
        df = load_dataset(cif_dir, id_prop_file, args.dataset, args.property)
        print(f"✅ 加载数据集: {len(df)} 样本")

        # 如果设置了max_samples，进行采样
        if args.max_samples and len(df) > args.max_samples:
            print(f"⚠️  数据集过大，随机采样 {args.max_samples} 样本")
            import random
            random.seed(42)
            df = random.sample(df, args.max_samples)

        # 创建数据加载器（使用本地数据）
        train_loader, val_loader, test_loader, _ = get_train_val_loaders(
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
            atom_features=config.atom_features if hasattr(config, 'atom_features') else 'cgcnn',
            neighbor_strategy='k-nearest',
            line_graph=config.line_graph if hasattr(config, 'line_graph') else True,
            split_seed=42,
            workers=0,
            pin_memory=False,
            save_dataloader=False,
            filename='temp_comparison',
            id_tag='jid',
            use_canonize=True,
            cutoff=8.0,
            max_neighbors=12,
            output_dir=args.save_dir
        )
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        print("请确保:")
        print(f"  1. 数据集路径正确: {args.root_dir}")
        print(f"  2. 数据集类型正确: {args.dataset}")
        print(f"  3. 属性名称正确: {args.property}")
        raise

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
