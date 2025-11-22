#!/usr/bin/env python
"""
诊断脚本：检查为什么不同模态配置下 graph_base 不一样
"""

import torch
import numpy as np
from compare_fusion_mechanisms import FusionMechanismComparator
import argparse

def diagnose_graph_base_difference(checkpoint1, checkpoint2, root_dir):
    """
    对比两个模型的 graph_base 特征是否相同
    """
    print("=" * 80)
    print("Graph_base 差异诊断")
    print("=" * 80)

    # 加载第一个模型
    print(f"\n加载模型 1: {checkpoint1}")
    comparator1 = FusionMechanismComparator(checkpoint1, root_dir)

    # 加载第二个模型
    print(f"加载模型 2: {checkpoint2}")
    comparator2 = FusionMechanismComparator(checkpoint2, root_dir)

    # 提取特征
    print("\n提取特征...")
    features1, targets1 = comparator1.extract_features()
    features2, targets2 = comparator2.extract_features()

    # 检查 graph_base
    if 'graph_base' not in features1 or 'graph_base' not in features2:
        print("❌ 错误：某个模型没有 graph_base 特征")
        return

    graph_base1 = features1['graph_base']
    graph_base2 = features2['graph_base']

    print(f"\n模型 1 graph_base shape: {graph_base1.shape}")
    print(f"模型 2 graph_base shape: {graph_base2.shape}")

    # 检查是否完全相同
    if np.allclose(graph_base1, graph_base2, rtol=1e-5, atol=1e-8):
        print("\n✅ graph_base 特征完全相同！")
        print("   差异可能来自 t-SNE 可视化的随机性（虽然设置了 random_state）")
    else:
        print("\n❌ graph_base 特征不同！")

        # 计算差异统计
        diff = np.abs(graph_base1 - graph_base2)
        print(f"\n差异统计:")
        print(f"  最大差异: {diff.max():.6f}")
        print(f"  平均差异: {diff.mean():.6f}")
        print(f"  差异标准差: {diff.std():.6f}")

        # 计算相关性
        from scipy.stats import pearsonr

        # 展平特征进行相关性分析
        flat1 = graph_base1.flatten()
        flat2 = graph_base2.flatten()
        corr, _ = pearsonr(flat1, flat2)

        print(f"\n  特征相关性: {corr:.6f}")

        if corr > 0.99:
            print("  → 特征高度相关，可能是训练过程中的微小差异")
        elif corr > 0.9:
            print("  → 特征较为相关，但有明显差异")
        else:
            print("  → 特征差异很大，很可能是不同的训练配置导致")

        print("\n可能的原因:")
        print("  1. 两个模型使用不同的训练配置训练（最可能）")
        print("  2. ALIGNN 层的权重不同")
        print("  3. BatchNorm 统计量不同")
        print("  4. 训练数据或顺序不同")

    # 检查目标值是否相同（确认数据一致）
    if np.allclose(targets1, targets2):
        print("\n✅ 目标值（targets）相同 - 使用了相同的数据")
    else:
        print("\n⚠️  目标值（targets）不同 - 可能使用了不同的数据或数据顺序")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="诊断 graph_base 差异")
    parser.add_argument('--checkpoint1', type=str, required=True, help='第一个模型 checkpoint')
    parser.add_argument('--checkpoint2', type=str, required=True, help='第二个模型 checkpoint')
    parser.add_argument('--root_dir', type=str, required=True, help='数据目录')

    args = parser.parse_args()

    diagnose_graph_base_difference(args.checkpoint1, args.checkpoint2, args.root_dir)
