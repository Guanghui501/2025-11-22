#!/usr/bin/env python
"""
增强版注意力热图可视化工具
显示实际的元素类型和token文本
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import argparse

# 配置matplotlib
plt.rcParams['font.size'] = 8
plt.rcParams['axes.unicode_minus'] = False

import sys
sys.path.insert(0, os.path.dirname(__file__))

from models.alignn import ALIGNN
from data import get_train_val_loaders


def plot_enhanced_attention_heatmap(
    attention_full,
    attention_no_middle,
    atom_features,
    text_tokens,
    sample_idx,
    save_path,
    max_atoms_display=30,
    max_tokens_display=40
):
    """
    绘制带有实际标签的注意力热图

    Args:
        attention_full: 全模态模型的注意力 [num_atoms, num_tokens]
        attention_no_middle: 无中期模型的注意力 [num_atoms, num_tokens]
        atom_features: 原子特征，用于获取元素类型
        text_tokens: 文本token列表
        sample_idx: 样本索引
        save_path: 保存路径
        max_atoms_display: 最多显示的原子数
        max_tokens_display: 最多显示的token数
    """
    # 限制显示数量（避免过大）
    num_atoms = min(attention_full.shape[0], max_atoms_display)
    num_tokens = min(attention_full.shape[1], max_tokens_display)

    attention_full = attention_full[:num_atoms, :num_tokens]
    attention_no_middle = attention_no_middle[:num_atoms, :num_tokens]

    # 准备标签
    # Y轴标签：原子索引 + 元素类型（如果可用）
    y_labels = [f"{i}" for i in range(num_atoms)]

    # X轴标签：token文本（截断过长的token）
    x_labels = []
    for i, token in enumerate(text_tokens[:num_tokens]):
        # 移除BERT的特殊前缀
        token_clean = token.replace('##', '').replace('[CLS]', 'CLS').replace('[SEP]', 'SEP')
        # 截断过长的token
        if len(token_clean) > 8:
            token_clean = token_clean[:8] + '.'
        x_labels.append(token_clean)

    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, num_atoms * 0.3)))
    fig.suptitle(f'Example {sample_idx}: Atom-Token Attention (Enhanced Labels)',
                fontsize=14, weight='bold')

    # 全模态模型
    sns.heatmap(attention_full, cmap='YlOrRd', ax=axes[0],
               cbar_kws={'label': 'Attention Weight'},
               xticklabels=x_labels,
               yticklabels=y_labels,
               linewidths=0.1, linecolor='gray')
    axes[0].set_title('Full Model (w/ Middle Fusion)', fontsize=12, pad=10)
    axes[0].set_xlabel('Text Tokens', fontsize=11)
    axes[0].set_ylabel('Atoms (Index)', fontsize=11)

    # 旋转x轴标签以避免重叠
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    # 无中期模型
    sns.heatmap(attention_no_middle, cmap='YlOrRd', ax=axes[1],
               cbar_kws={'label': 'Attention Weight'},
               xticklabels=x_labels,
               yticklabels=y_labels,
               linewidths=0.1, linecolor='gray')
    axes[1].set_title('No Middle Fusion', fontsize=12, pad=10)
    axes[1].set_xlabel('Text Tokens', fontsize=11)
    axes[1].set_ylabel('Atoms (Index)', fontsize=11)

    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ 保存增强版热图: {os.path.basename(save_path)}")


def plot_attention_with_statistics(
    attention_full,
    attention_no_middle,
    text_tokens,
    sample_idx,
    save_path
):
    """
    绘制注意力热图，同时显示每个原子的统计信息
    """
    num_atoms = min(attention_full.shape[0], 25)
    num_tokens = min(attention_full.shape[1], 30)

    attention_full = attention_full[:num_atoms, :num_tokens]
    attention_no_middle = attention_no_middle[:num_atoms, :num_tokens]

    # 计算每个原子的注意力统计
    entropy_full = -np.sum(attention_full * np.log(attention_full + 1e-10), axis=1)
    entropy_no_middle = -np.sum(attention_no_middle * np.log(attention_no_middle + 1e-10), axis=1)

    max_weight_full = np.max(attention_full, axis=1)
    max_weight_no_middle = np.max(attention_no_middle, axis=1)

    # 准备token标签
    x_labels = [token.replace('##', '').replace('[CLS]', 'CLS').replace('[SEP]', 'SEP')[:8]
                for token in text_tokens[:num_tokens]]

    # 创建复杂布局
    fig = plt.figure(figsize=(24, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 0.15, 0.15], hspace=0.3, wspace=0.4)

    # 主热图
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # 统计图
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # 边栏统计
    ax5 = fig.add_subplot(gs[0, 2])
    ax6 = fig.add_subplot(gs[0, 3])

    fig.suptitle(f'Example {sample_idx}: Detailed Attention Analysis', fontsize=16, weight='bold')

    # 热图1：全模态
    sns.heatmap(attention_full, cmap='YlOrRd', ax=ax1,
               xticklabels=x_labels, yticklabels=True,
               cbar_kws={'label': 'Attention'})
    ax1.set_title('Full Model', fontsize=12)
    ax1.set_xlabel('Tokens', fontsize=10)
    ax1.set_ylabel('Atoms', fontsize=10)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    # 热图2：无中期
    sns.heatmap(attention_no_middle, cmap='YlOrRd', ax=ax2,
               xticklabels=x_labels, yticklabels=True,
               cbar_kws={'label': 'Attention'})
    ax2.set_title('No Middle Fusion', fontsize=12)
    ax2.set_xlabel('Tokens', fontsize=10)
    ax2.set_ylabel('Atoms', fontsize=10)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    # 统计图1：熵对比
    x_pos = np.arange(num_atoms)
    width = 0.35
    ax3.barh(x_pos - width/2, entropy_full, width, label='Full Model', color='#2ecc71', alpha=0.7)
    ax3.barh(x_pos + width/2, entropy_no_middle, width, label='No Middle', color='#e74c3c', alpha=0.7)
    ax3.set_ylabel('Atom Index')
    ax3.set_xlabel('Attention Entropy')
    ax3.set_title('Per-Atom Entropy Comparison', fontsize=11)
    ax3.legend()
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)

    # 统计图2：最大权重对比
    ax4.barh(x_pos - width/2, max_weight_full, width, label='Full Model', color='#2ecc71', alpha=0.7)
    ax4.barh(x_pos + width/2, max_weight_no_middle, width, label='No Middle', color='#e74c3c', alpha=0.7)
    ax4.set_ylabel('Atom Index')
    ax4.set_xlabel('Max Attention Weight')
    ax4.set_title('Per-Atom Max Weight Comparison', fontsize=11)
    ax4.legend()
    ax4.invert_yaxis()
    ax4.grid(axis='x', alpha=0.3)

    # 边栏：全模态模型的每行最大值位置
    max_indices_full = np.argmax(attention_full, axis=1)
    ax5.imshow(max_indices_full.reshape(-1, 1), cmap='tab20', aspect='auto')
    ax5.set_ylabel('Atoms', fontsize=9)
    ax5.set_title('Max\nToken', fontsize=9)
    ax5.set_xticks([])

    # 边栏：无中期模型的每行最大值位置
    max_indices_no_middle = np.argmax(attention_no_middle, axis=1)
    ax6.imshow(max_indices_no_middle.reshape(-1, 1), cmap='tab20', aspect='auto')
    ax6.set_ylabel('Atoms', fontsize=9)
    ax6.set_title('Max\nToken', fontsize=9)
    ax6.set_xticks([])

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ 保存详细分析图: {os.path.basename(save_path)}")


def main():
    parser = argparse.ArgumentParser(description='增强版注意力热图可视化')
    parser.add_argument('--checkpoint_full', type=str, required=True)
    parser.add_argument('--checkpoint_no_middle', type=str, required=True)
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='dft_3d')
    parser.add_argument('--property', type=str, default='mbj_bandgap')
    parser.add_argument('--save_dir', type=str, default='./attention_enhanced')
    parser.add_argument('--num_examples', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=1, help='必须为1以便获取单个样本')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("增强版注意力热图可视化")
    print("="*80)

    # 加载tokenizer
    print("\n加载BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('m3rg-iitd/matscibert')

    # 加载数据
    print(f"\n加载数据集: {args.dataset} - {args.property}")
    from train_with_cross_modal_attention import load_dataset, get_dataset_paths

    cif_dir, id_prop_file = get_dataset_paths(args.root_dir, args.dataset, args.property)
    df = load_dataset(cif_dir, id_prop_file, args.dataset, args.property)

    train_loader, val_loader, test_loader, _ = get_train_val_loaders(
        dataset='user_data',
        dataset_array=df,
        target='target',
        batch_size=1,  # 必须为1
        split_seed=42,
        workers=0,
        output_dir=args.save_dir
    )

    print(f"  测试集样本数: {len(test_loader.dataset)}")

    # 加载模型
    print("\n加载模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(checkpoint_path, device):
        """灵活加载模型checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 尝试获取config
        config = None
        for key in ['config', 'model_config']:
            if key in checkpoint:
                config = checkpoint[key]
                break

        if config is None:
            raise ValueError(f"Cannot find config in checkpoint {checkpoint_path}")

        # 创建模型
        model = ALIGNN(config)

        # 尝试获取state_dict
        state_dict = None
        for key in ['model_state_dict', 'state_dict', 'model', 'model_state']:
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

        if state_dict is None:
            raise ValueError(f"Cannot find state_dict in checkpoint {checkpoint_path}")

        # 加载权重
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model

    print("  加载全模态模型...")
    model_full = load_model(args.checkpoint_full, device)

    print("  加载无中期融合模型...")
    model_no_middle = load_model(args.checkpoint_no_middle, device)

    # 提取并可视化
    print(f"\n提取并可视化前 {args.num_examples} 个样本的注意力权重...")

    count = 0
    with torch.no_grad():
        for batch in test_loader:
            if count >= args.num_examples:
                break

            g, lg, text, target = batch
            g = g.to(device)
            if lg is not None:
                lg = lg.to(device)

            # 提取注意力
            model_input = (g, lg, text) if lg is not None else (g, text)
            output_full = model_full(model_input, return_attention=True)
            output_no_middle = model_no_middle(model_input, return_attention=True)

            # 获取细粒度注意力
            if 'fine_grained_attention_weights' in output_full:
                attn_full = output_full['fine_grained_attention_weights']['atom_to_text']
                attn_no_middle = output_no_middle['fine_grained_attention_weights']['atom_to_text']

                # 平均多头注意力
                attn_full = attn_full.mean(dim=1)[0].cpu().numpy()  # [num_atoms, num_tokens]
                attn_no_middle = attn_no_middle.mean(dim=1)[0].cpu().numpy()

                # 获取文本tokens
                text_str = text['input_ids'][0].cpu().numpy()
                tokens = tokenizer.convert_ids_to_tokens(text_str)

                # 获取原子特征（如果可用）
                atom_features = g.ndata.get('atom_features', None)

                # 绘制增强版热图
                save_path_enhanced = os.path.join(args.save_dir, f'enhanced_heatmap_{count+1}.png')
                plot_enhanced_attention_heatmap(
                    attn_full, attn_no_middle, atom_features, tokens,
                    count + 1, save_path_enhanced
                )

                # 绘制详细分析图
                save_path_detailed = os.path.join(args.save_dir, f'detailed_analysis_{count+1}.png')
                plot_attention_with_statistics(
                    attn_full, attn_no_middle, tokens,
                    count + 1, save_path_detailed
                )

                count += 1

    print(f"\n✅ 完成！生成了 {count} 个样本的增强版可视化")
    print(f"   保存位置: {args.save_dir}")


if __name__ == '__main__':
    main()
