#!/usr/bin/env python
"""
可视化多模态融合机制的特征流程
生成融合机制的流程图和特征维度变化图
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9


def draw_fusion_timeline():
    """绘制融合机制的时间线"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # 标题
    ax.text(5, 11.5, 'Multi-Modal Fusion Mechanisms Timeline',
            ha='center', fontsize=16, weight='bold')

    # ==================== 图特征流 ====================
    # 标签
    ax.text(0.5, 10, 'Graph\nFeatures', ha='center', va='center',
            fontsize=11, weight='bold', color='#2E86AB')

    # 1. 原子嵌入
    box1 = FancyBboxPatch((0.2, 9.2), 0.6, 0.6, boxstyle="round,pad=0.05",
                          facecolor='#A8DADC', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(box1)
    ax.text(0.5, 9.5, 'Atom\nEmbed', ha='center', va='center', fontsize=8)
    ax.text(0.5, 8.9, '[N,92]→[N,256]', ha='center', fontsize=7, style='italic')

    # 箭头到ALIGNN
    ax.annotate('', xy=(1.5, 9.5), xytext=(0.8, 9.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2E86AB'))

    # 2. ALIGNN层 (循环)
    alignn_box = FancyBboxPatch((1.5, 8.5), 1.2, 1.8, boxstyle="round,pad=0.1",
                                facecolor='#A8DADC', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(alignn_box)
    ax.text(2.1, 10, 'ALIGNN Layers (x4)', ha='center', fontsize=9, weight='bold')

    # 显示中间融合位置
    ax.text(2.1, 9.5, 'Layer 0', ha='center', fontsize=7)
    ax.text(2.1, 9.2, 'Layer 1', ha='center', fontsize=7)

    # 中间融合标记
    fusion_mark = FancyBboxPatch((1.6, 8.85), 1.0, 0.25, boxstyle="round,pad=0.02",
                                 facecolor='#F4A261', edgecolor='#E76F51', linewidth=1.5)
    ax.add_patch(fusion_mark)
    ax.text(2.1, 8.97, 'Layer 2 + Middle Fusion', ha='center', fontsize=7, weight='bold')

    ax.text(2.1, 8.7, 'Layer 3', ha='center', fontsize=7)
    ax.text(2.1, 8.3, '[N,256]', ha='center', fontsize=7, style='italic')

    # 箭头到GCN
    ax.annotate('', xy=(3.3, 9.4), xytext=(2.7, 9.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2E86AB'))

    # 3. GCN层
    gcn_box = FancyBboxPatch((3.3, 8.9), 0.8, 1.0, boxstyle="round,pad=0.05",
                             facecolor='#A8DADC', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(gcn_box)
    ax.text(3.7, 9.5, 'GCN\nLayers\n(x4)', ha='center', va='center', fontsize=8)
    ax.text(3.7, 8.7, '[N,256]', ha='center', fontsize=7, style='italic')

    # 箭头到细粒度注意力
    ax.annotate('', xy=(4.7, 9.4), xytext=(4.1, 9.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2E86AB'))

    # 4. 细粒度注意力
    fine_box = FancyBboxPatch((4.7, 8.7), 1.0, 1.4, boxstyle="round,pad=0.05",
                              facecolor='#F4A261', edgecolor='#E76F51', linewidth=2)
    ax.add_patch(fine_box)
    ax.text(5.2, 9.7, 'Fine-grained\nAttention', ha='center', fontsize=8, weight='bold')
    ax.text(5.2, 9.2, 'Atom↔Token', ha='center', fontsize=7)
    ax.text(5.2, 8.9, '[B,M,256]', ha='center', fontsize=7, style='italic')

    # 箭头到池化
    ax.annotate('', xy=(6.3, 9.4), xytext=(5.7, 9.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2E86AB'))

    # 5. 图池化
    pool_box = FancyBboxPatch((6.3, 9.0), 0.7, 0.8, boxstyle="round,pad=0.05",
                              facecolor='#A8DADC', edgecolor='#2E86AB', linewidth=2)
    ax.add_patch(pool_box)
    ax.text(6.65, 9.5, 'Readout\n+Proj', ha='center', va='center', fontsize=8)
    ax.text(6.65, 8.8, '[B,64]', ha='center', fontsize=7, style='italic')

    # 箭头到全局注意力
    ax.annotate('', xy=(7.6, 9.4), xytext=(7.0, 9.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2E86AB'))

    # 6. 全局注意力
    global_box = FancyBboxPatch((7.6, 8.5), 1.0, 1.8, boxstyle="round,pad=0.05",
                                facecolor='#F4A261', edgecolor='#E76F51', linewidth=2)
    ax.add_patch(global_box)
    ax.text(8.1, 9.8, 'Cross-modal\nAttention', ha='center', fontsize=8, weight='bold')
    ax.text(8.1, 9.2, 'Graph↔Text', ha='center', fontsize=7)
    ax.text(8.1, 8.9, '[B,64]', ha='center', fontsize=7, style='italic')

    # 箭头到融合
    ax.annotate('', xy=(9.2, 9.4), xytext=(8.6, 9.4),
                arrowprops=dict(arrowstyle='->', lw=2, color='#2E86AB'))

    # 7. 最终融合
    final_box = FancyBboxPatch((9.2, 9.0), 0.6, 0.8, boxstyle="round,pad=0.05",
                               facecolor='#6A4C93', edgecolor='#553285', linewidth=2)
    ax.add_patch(final_box)
    ax.text(9.5, 9.5, 'Fusion\n+FC', ha='center', va='center', fontsize=8, color='white')
    ax.text(9.5, 8.8, '[B,1]', ha='center', fontsize=7, style='italic', color='white')

    # ==================== 文本特征流 ====================
    # 标签
    ax.text(0.5, 6.5, 'Text\nFeatures', ha='center', va='center',
            fontsize=11, weight='bold', color='#E76F51')

    # 1. MatSciBERT
    bert_box = FancyBboxPatch((0.2, 5.5), 0.6, 0.8, boxstyle="round,pad=0.05",
                              facecolor='#FFDAB9', edgecolor='#E76F51', linewidth=2)
    ax.add_patch(bert_box)
    ax.text(0.5, 6.0, 'BERT\nEncode', ha='center', va='center', fontsize=8)
    ax.text(0.5, 5.3, '[B,L,768]', ha='center', fontsize=7, style='italic')

    # 箭头
    ax.annotate('', xy=(1.5, 5.9), xytext=(0.8, 5.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='#E76F51'))

    # 2. CLS + 投影
    proj_box = FancyBboxPatch((1.5, 5.5), 0.8, 0.8, boxstyle="round,pad=0.05",
                              facecolor='#FFDAB9', edgecolor='#E76F51', linewidth=2)
    ax.add_patch(proj_box)
    ax.text(1.9, 6.0, 'CLS\n+Proj', ha='center', va='center', fontsize=8)
    ax.text(1.9, 5.3, '[B,64]', ha='center', fontsize=7, style='italic')

    # 中间融合箭头（向上到ALIGNN）
    ax.annotate('', xy=(2.1, 8.85), xytext=(1.9, 6.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='#F4A261',
                               linestyle='dashed', alpha=0.7))
    ax.text(1.3, 7.5, 'Middle\nFusion', ha='center', fontsize=7,
            color='#E76F51', weight='bold', rotation=70)

    # 文本token保持（用于细粒度注意力）
    ax.annotate('', xy=(5.2, 8.7), xytext=(1.5, 5.7),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#F4A261',
                               linestyle='dashed', alpha=0.5))
    ax.text(3.0, 7.0, 'Text Tokens [B,L,768]', ha='center', fontsize=7,
            color='#E76F51', style='italic', rotation=15)

    # 文本特征到全局注意力
    ax.annotate('', xy=(8.1, 8.5), xytext=(2.3, 5.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='#E76F51'))

    # 文本增强路径（从细粒度注意力）
    ax.annotate('', xy=(8.1, 8.7), xytext=(5.7, 9.0),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#F4A261',
                               linestyle='dashed', alpha=0.7))

    # 到最终融合
    ax.annotate('', xy=(9.5, 9.0), xytext=(8.6, 8.7),
                arrowprops=dict(arrowstyle='->', lw=2, color='#E76F51'))

    # ==================== 对比学习（底部）====================
    contrast_box = FancyBboxPatch((3.5, 1.5), 3.0, 1.2, boxstyle="round,pad=0.1",
                                  facecolor='#FFE5B4', edgecolor='#DAA520', linewidth=2)
    ax.add_patch(contrast_box)
    ax.text(5.0, 2.4, 'Contrastive Learning Loss', ha='center', fontsize=10, weight='bold')
    ax.text(5.0, 2.0, 'Align graph and text in feature space', ha='center', fontsize=8)
    ax.text(5.0, 1.7, '(Training time only, affects feature distribution)', ha='center',
            fontsize=7, style='italic')

    # 对比学习连接
    ax.annotate('', xy=(5.0, 1.5), xytext=(6.65, 8.8),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='#DAA520',
                               linestyle='dotted', alpha=0.6))
    ax.annotate('', xy=(5.0, 1.5), xytext=(1.9, 5.3),
                arrowprops=dict(arrowstyle='<->', lw=1.5, color='#DAA520',
                               linestyle='dotted', alpha=0.6))

    # ==================== 图例 ====================
    legend_y = 0.5

    # 图特征
    graph_patch = mpatches.Patch(facecolor='#A8DADC', edgecolor='#2E86AB',
                                 linewidth=2, label='Graph Processing')
    # 文本特征
    text_patch = mpatches.Patch(facecolor='#FFDAB9', edgecolor='#E76F51',
                                linewidth=2, label='Text Processing')
    # 融合模块
    fusion_patch = mpatches.Patch(facecolor='#F4A261', edgecolor='#E76F51',
                                  linewidth=2, label='Fusion Module')
    # 最终预测
    final_patch = mpatches.Patch(facecolor='#6A4C93', edgecolor='#553285',
                                 linewidth=2, label='Final Prediction')
    # 对比学习
    contrast_patch = mpatches.Patch(facecolor='#FFE5B4', edgecolor='#DAA520',
                                    linewidth=2, label='Contrastive Learning')

    ax.legend(handles=[graph_patch, text_patch, fusion_patch, final_patch, contrast_patch],
              loc='lower left', fontsize=8, ncol=5)

    # 添加注释
    ax.text(5.0, 0.2, 'N: total atoms | B: batch size | L: sequence length | M: max atoms',
            ha='center', fontsize=7, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig('fusion_mechanisms_timeline.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('fusion_mechanisms_timeline.png', dpi=300, bbox_inches='tight')
    print("✅ 保存: fusion_mechanisms_timeline.pdf/png")
    plt.close()


def draw_feature_dimensions():
    """绘制特征维度变化图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    mechanisms = [
        {
            'title': '1. Contrastive Learning',
            'ax': axes[0, 0],
            'stages': ['Original\nGraph', 'L2 Norm', 'After\nContrastive'],
            'graph_dims': [64, 64, 64],
            'text_dims': [64, 64, 64],
            'graph_color': ['#A8DADC', '#7FB3D5', '#5499C7'],
            'text_color': ['#FFDAB9', '#FFCBA4', '#FFAC86'],
            'effect': 'Aligns features in\nsemantic space'
        },
        {
            'title': '2. Middle Fusion',
            'ax': axes[0, 1],
            'stages': ['ALIGNN\nLayer 1', 'Middle\nFusion', 'ALIGNN\nLayer 3'],
            'graph_dims': [256, 256, 256],
            'text_dims': [None, 64, None],
            'graph_color': ['#A8DADC', '#F4A261', '#A8DADC'],
            'text_color': [None, '#FFDAB9', None],
            'effect': 'Text modulates\ngraph encoding'
        },
        {
            'title': '3. Fine-grained Attention',
            'ax': axes[1, 0],
            'stages': ['Node Feat\n[B,M,256]', 'Attention', 'Enhanced\nNodes'],
            'graph_dims': [256, 256, 256],
            'text_dims': [768, 768, 768],
            'graph_color': ['#A8DADC', '#F4A261', '#E76F51'],
            'text_color': ['#FFDAB9', '#F4A261', '#E76F51'],
            'effect': 'Atom-token level\nbidirectional fusion'
        },
        {
            'title': '4. Global Cross-modal Attention',
            'ax': axes[1, 1],
            'stages': ['Graph\nProj', 'Attention', 'Fusion'],
            'graph_dims': [64, 64, 64],
            'text_dims': [64, 64, 64],
            'graph_color': ['#A8DADC', '#F4A261', '#6A4C93'],
            'text_color': ['#FFDAB9', '#F4A261', '#6A4C93'],
            'effect': 'Global bidirectional\nfusion'
        }
    ]

    for mech in mechanisms:
        ax = mech['ax']
        stages = mech['stages']
        n_stages = len(stages)

        x_pos = np.arange(n_stages)
        width = 0.35

        # 绘制图特征条形
        graph_bars = ax.bar(x_pos - width/2, mech['graph_dims'], width,
                           label='Graph Features', color=mech['graph_color'],
                           edgecolor='black', linewidth=1.5)

        # 绘制文本特征条形
        text_dims_plot = [0 if d is None else d for d in mech['text_dims']]
        text_bars = ax.bar(x_pos + width/2, text_dims_plot, width,
                          label='Text Features',
                          color=[c if c else 'white' for c in mech['text_color']],
                          edgecolor='black', linewidth=1.5)

        # 标注维度
        for i, (g_dim, t_dim) in enumerate(zip(mech['graph_dims'], mech['text_dims'])):
            if g_dim:
                ax.text(i - width/2, g_dim + 10, str(g_dim),
                       ha='center', va='bottom', fontsize=9, weight='bold')
            if t_dim:
                ax.text(i + width/2, t_dim + 10, str(t_dim),
                       ha='center', va='bottom', fontsize=9, weight='bold')

        ax.set_ylabel('Feature Dimension', fontsize=10)
        ax.set_title(mech['title'], fontsize=12, weight='bold', pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stages, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(0, max(mech['graph_dims']) * 1.2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加效果说明
        ax.text(0.5, -0.25, mech['effect'], transform=ax.transAxes,
               ha='center', va='top', fontsize=9, style='italic',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Feature Dimension Changes in Different Fusion Mechanisms',
                fontsize=14, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('feature_dimensions_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('feature_dimensions_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ 保存: feature_dimensions_comparison.pdf/png")
    plt.close()


def draw_attention_heatmap_example():
    """绘制细粒度注意力的示例热图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 示例数据
    atoms = ['Si₁', 'O₁', 'O₂', 'Si₂', 'O₃', 'O₄']
    tokens = ['Silicon', 'dioxide', 'with', 'high', 'thermal', 'stability']

    # 原子→文本注意力
    atom_to_text = np.array([
        [0.7, 0.2, 0.0, 0.0, 0.1, 0.0],  # Si₁ → Silicon
        [0.1, 0.6, 0.0, 0.0, 0.2, 0.1],  # O₁ → dioxide, thermal
        [0.1, 0.6, 0.0, 0.0, 0.2, 0.1],  # O₂ → dioxide, thermal
        [0.7, 0.2, 0.0, 0.0, 0.1, 0.0],  # Si₂ → Silicon
        [0.1, 0.5, 0.0, 0.0, 0.3, 0.1],  # O₃ → dioxide, thermal
        [0.1, 0.5, 0.0, 0.0, 0.3, 0.1],  # O₄ → dioxide, thermal
    ])

    # 文本→原子注意力
    text_to_atom = atom_to_text.T

    # 绘制原子→文本
    im1 = axes[0].imshow(atom_to_text, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(len(tokens)))
    axes[0].set_yticks(np.arange(len(atoms)))
    axes[0].set_xticklabels(tokens, fontsize=10)
    axes[0].set_yticklabels(atoms, fontsize=10)
    axes[0].set_xlabel('Text Tokens', fontsize=11, weight='bold')
    axes[0].set_ylabel('Atoms', fontsize=11, weight='bold')
    axes[0].set_title('Atom-to-Token Attention\n(Which words does each atom attend to?)',
                     fontsize=12, weight='bold', pad=10)

    # 添加数值标注
    for i in range(len(atoms)):
        for j in range(len(tokens)):
            if atom_to_text[i, j] > 0.05:
                text = axes[0].text(j, i, f'{atom_to_text[i, j]:.1f}',
                                   ha="center", va="center", color="white" if atom_to_text[i, j] > 0.4 else "black",
                                   fontsize=8, weight='bold')

    plt.colorbar(im1, ax=axes[0], label='Attention Weight')

    # 绘制文本→原子
    im2 = axes[1].imshow(text_to_atom, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(len(atoms)))
    axes[1].set_yticks(np.arange(len(tokens)))
    axes[1].set_xticklabels(atoms, fontsize=10)
    axes[1].set_yticklabels(tokens, fontsize=10)
    axes[1].set_xlabel('Atoms', fontsize=11, weight='bold')
    axes[1].set_ylabel('Text Tokens', fontsize=11, weight='bold')
    axes[1].set_title('Token-to-Atom Attention\n(Which atoms does each word attend to?)',
                     fontsize=12, weight='bold', pad=10)

    # 添加数值标注
    for i in range(len(tokens)):
        for j in range(len(atoms)):
            if text_to_atom[i, j] > 0.05:
                text = axes[1].text(j, i, f'{text_to_atom[i, j]:.1f}',
                                   ha="center", va="center", color="white" if text_to_atom[i, j] > 0.4 else "black",
                                   fontsize=8, weight='bold')

    plt.colorbar(im2, ax=axes[1], label='Attention Weight')

    plt.suptitle('Fine-grained Cross-modal Attention Example\n(SiO₂ with thermal properties)',
                fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('attention_heatmap_example.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('attention_heatmap_example.png', dpi=300, bbox_inches='tight')
    print("✅ 保存: attention_heatmap_example.pdf/png")
    plt.close()


if __name__ == '__main__':
    print("🎨 生成多模态融合机制可视化...")
    print()

    print("1️⃣ 绘制融合机制时间线...")
    draw_fusion_timeline()

    print("2️⃣ 绘制特征维度变化对比...")
    draw_feature_dimensions()

    print("3️⃣ 绘制细粒度注意力示例...")
    draw_attention_heatmap_example()

    print()
    print("=" * 60)
    print("✅ 所有可视化图片已生成！")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - fusion_mechanisms_timeline.pdf/png")
    print("  - feature_dimensions_comparison.pdf/png")
    print("  - attention_heatmap_example.pdf/png")
