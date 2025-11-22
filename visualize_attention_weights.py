#!/usr/bin/env python
"""
可视化细粒度注意力权重
展示原子-文本词之间的对应关系
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from models.alignn import ALIGNN
from data import get_train_val_loaders

sns.set_style("white")
plt.rcParams['font.size'] = 9


def visualize_attention_heatmap(attn_weights, atoms, tokens, title, save_path,
                                cmap='YlOrRd', figsize=(12, 8)):
    """
    绘制注意力权重热图

    Args:
        attn_weights: [num_atoms, num_tokens] 注意力权重矩阵
        atoms: 原子列表
        tokens: 文本token列表
        title: 标题
        save_path: 保存路径
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热图
    im = ax.imshow(attn_weights, cmap=cmap, aspect='auto', vmin=0, vmax=1)

    # 设置坐标轴
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(atoms)))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(atoms, fontsize=10)

    ax.set_xlabel('Text Tokens', fontsize=12, weight='bold')
    ax.set_ylabel('Atoms', fontsize=12, weight='bold')
    ax.set_title(title, fontsize=14, weight='bold', pad=15)

    # 添加数值标注（只显示权重>0.05的）
    for i in range(len(atoms)):
        for j in range(len(tokens)):
            if attn_weights[i, j] > 0.05:
                text_color = 'white' if attn_weights[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{attn_weights[i, j]:.2f}',
                       ha="center", va="center", color=text_color,
                       fontsize=8, weight='bold')

    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存: {save_path}")
    plt.close()


def visualize_bidirectional_attention(atom_to_text, text_to_atom, atoms, tokens,
                                     material_name, save_path):
    """
    并排显示双向注意力

    Args:
        atom_to_text: [num_atoms, num_tokens] 原子→文本注意力
        text_to_atom: [num_tokens, num_atoms] 文本→原子注意力
        atoms: 原子列表
        tokens: 文本token列表
        material_name: 材料名称
        save_path: 保存路径
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 左: 原子→文本
    im1 = ax1.imshow(atom_to_text, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(np.arange(len(tokens)))
    ax1.set_yticks(np.arange(len(atoms)))
    ax1.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax1.set_yticklabels(atoms, fontsize=9)
    ax1.set_xlabel('Text Tokens', fontsize=11, weight='bold')
    ax1.set_ylabel('Atoms', fontsize=11, weight='bold')
    ax1.set_title('Atom → Token Attention\n(Which words does each atom focus on?)',
                 fontsize=12, weight='bold', pad=10)

    # 标注
    for i in range(len(atoms)):
        for j in range(len(tokens)):
            if atom_to_text[i, j] > 0.05:
                color = 'white' if atom_to_text[i, j] > 0.5 else 'black'
                ax1.text(j, i, f'{atom_to_text[i, j]:.2f}',
                        ha="center", va="center", color=color, fontsize=7)

    plt.colorbar(im1, ax=ax1, label='Attention Weight')

    # 右: 文本→原子
    im2 = ax2.imshow(text_to_atom, cmap='YlGnBu', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(np.arange(len(atoms)))
    ax2.set_yticks(np.arange(len(tokens)))
    ax2.set_xticklabels(atoms, rotation=45, ha='right', fontsize=9)
    ax2.set_yticklabels(tokens, fontsize=9)
    ax2.set_xlabel('Atoms', fontsize=11, weight='bold')
    ax2.set_ylabel('Text Tokens', fontsize=11, weight='bold')
    ax2.set_title('Token → Atom Attention\n(Which atoms does each word focus on?)',
                 fontsize=12, weight='bold', pad=10)

    # 标注
    for i in range(len(tokens)):
        for j in range(len(atoms)):
            if text_to_atom[i, j] > 0.05:
                color = 'white' if text_to_atom[i, j] > 0.5 else 'black'
                ax2.text(j, i, f'{text_to_atom[i, j]:.2f}',
                        ha="center", va="center", color=color, fontsize=7)

    plt.colorbar(im2, ax=ax2, label='Attention Weight')

    plt.suptitle(f'Fine-grained Cross-modal Attention: {material_name}',
                fontsize=14, weight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"  ✅ 保存双向注意力图: {save_path}")
    plt.close()


def extract_attention_examples(model, data_loader, device='cpu', num_examples=5):
    """
    提取若干个样本的注意力权重用于可视化

    Returns:
        examples: list of dict, each containing:
            - 'material_id': 材料ID
            - 'text': 文本描述
            - 'tokens': token列表
            - 'atoms': 原子符号列表
            - 'atom_to_text': [num_atoms, num_tokens]
            - 'text_to_atom': [num_tokens, num_atoms]
    """
    print(f"🔄 提取{num_examples}个样本的注意力权重...")

    model = model.to(device)
    model.eval()

    examples = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="提取注意力")):
            if len(examples) >= num_examples:
                break

            if len(batch) == 3:
                g, text, target = batch
                lg = None
            elif len(batch) == 4:
                g, lg, text, target = batch
            else:
                raise ValueError(f"不支持的batch格式")

            g = g.to(device)
            if lg is not None:
                lg = lg.to(device)

            batch_size = len(text) if isinstance(text, list) else text.size(0)

            # 构建模型输入
            if lg is not None:
                model_input = (g, lg, text)
            else:
                model_input = (g, text)

            # Forward with attention
            output = model(model_input, return_attention=True)

            if 'fine_grained_attention_weights' not in output:
                print("⚠️  模型未启用细粒度注意力，无法提取注意力权重")
                return []

            # 获取注意力权重
            attn_weights = output['fine_grained_attention_weights']
            atom_to_text_attn = attn_weights['atom_to_text']  # [batch, heads, atoms, tokens]
            text_to_atom_attn = attn_weights['text_to_atom']  # [batch, heads, tokens, atoms]

            # 对每个样本进行处理
            batch_num_nodes = g.batch_num_nodes()

            for i in range(min(batch_size, num_examples - len(examples))):
                # 获取该样本的文本
                text_str = text[i] if isinstance(text, list) else None

                if text_str is None:
                    continue

                # 从tokenizer获取tokens（简化：直接分词）
                tokens = text_str.split()[:20]  # 限制token数量

                # 获取原子信息
                num_atoms = batch_num_nodes[i].item()

                # 获取原子符号（简化：使用索引）
                atoms = [f"Atom{j+1}" for j in range(min(num_atoms, 30))]  # 限制原子数量

                # 平均多头注意力
                atom_to_text = atom_to_text_attn[i].mean(dim=0)  # [atoms, tokens]
                text_to_atom = text_to_atom_attn[i].mean(dim=0)  # [tokens, atoms]

                # 截取对应大小
                atom_to_text = atom_to_text[:len(atoms), :len(tokens)].cpu().numpy()
                text_to_atom = text_to_atom[:len(tokens), :len(atoms)].cpu().numpy()

                examples.append({
                    'material_id': f'Sample_{batch_idx}_{i}',
                    'text': text_str,
                    'tokens': tokens,
                    'atoms': atoms,
                    'atom_to_text': atom_to_text,
                    'text_to_atom': text_to_atom
                })

    print(f"✅ 提取了{len(examples)}个样本的注意力权重")
    return examples


def analyze_attention_patterns(examples, save_dir):
    """
    分析注意力模式

    统计:
    - 平均注意力权重分布
    - 高权重的原子-词对
    - 注意力稀疏度
    """
    print("\n📊 分析注意力模式...")

    report_path = os.path.join(save_dir, 'attention_analysis.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("细粒度注意力权重分析报告\n")
        f.write("="*80 + "\n\n")

        for idx, example in enumerate(examples):
            f.write(f"\n## 样本 {idx+1}: {example['material_id']}\n")
            f.write(f"文本描述: {example['text'][:100]}...\n\n")

            atom_to_text = example['atom_to_text']
            text_to_atom = example['text_to_atom']

            # 统计
            f.write(f"原子数: {len(example['atoms'])}\n")
            f.write(f"Token数: {len(example['tokens'])}\n")
            f.write(f"平均注意力权重: {atom_to_text.mean():.4f}\n")
            f.write(f"最大注意力权重: {atom_to_text.max():.4f}\n")
            f.write(f"稀疏度 (权重<0.1的比例): {(atom_to_text < 0.1).mean():.2%}\n\n")

            # 找出top-5高权重的原子-词对
            f.write("Top-5 原子-词注意力对:\n")
            flat_indices = np.argsort(atom_to_text.flatten())[::-1][:5]
            for rank, flat_idx in enumerate(flat_indices, 1):
                i, j = np.unravel_index(flat_idx, atom_to_text.shape)
                weight = atom_to_text[i, j]
                atom = example['atoms'][i]
                token = example['tokens'][j]
                f.write(f"  {rank}. {atom} ← {token}: {weight:.4f}\n")

            f.write("\n" + "-"*80 + "\n")

    print(f"✅ 分析报告已保存: {report_path}")


def visualize_attention_distribution(examples, save_dir):
    """
    可视化注意力权重分布
    """
    print("\n📊 可视化注意力权重分布...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 收集所有注意力权重
    all_weights = []
    for example in examples:
        all_weights.extend(example['atom_to_text'].flatten())

    # 1. 直方图
    ax = axes[0, 0]
    ax.hist(all_weights, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.set_xlabel('Attention Weight', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Attention Weights', fontsize=12, weight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 2. CDF
    ax = axes[0, 1]
    sorted_weights = np.sort(all_weights)
    cdf = np.arange(1, len(sorted_weights)+1) / len(sorted_weights)
    ax.plot(sorted_weights, cdf, linewidth=2, color='coral')
    ax.set_xlabel('Attention Weight', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('Cumulative Distribution Function', fontsize=12, weight='bold')
    ax.grid(alpha=0.3)

    # 3. Boxplot（每个样本）
    ax = axes[1, 0]
    box_data = [example['atom_to_text'].flatten() for example in examples]
    bp = ax.boxplot(box_data, labels=[f"S{i+1}" for i in range(len(examples))],
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
    ax.set_xlabel('Sample', fontsize=11)
    ax.set_ylabel('Attention Weight', fontsize=11)
    ax.set_title('Attention Weight Distribution per Sample', fontsize=12, weight='bold')
    ax.grid(axis='y', alpha=0.3)

    # 4. 统计表
    ax = axes[1, 1]
    ax.axis('off')

    stats_data = [
        ['Metric', 'Value'],
        ['Mean', f'{np.mean(all_weights):.4f}'],
        ['Median', f'{np.median(all_weights):.4f}'],
        ['Std Dev', f'{np.std(all_weights):.4f}'],
        ['Min', f'{np.min(all_weights):.4f}'],
        ['Max', f'{np.max(all_weights):.4f}'],
        ['Sparsity (<0.1)', f'{(np.array(all_weights) < 0.1).mean():.2%}'],
        ['Num Samples', f'{len(examples)}'],
    ]

    table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                    colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 交替行颜色
    for i in range(1, len(stats_data)):
        for j in range(2):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#E8F5E9')

    plt.suptitle('Attention Weight Statistics', fontsize=14, weight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'attention_distribution.pdf')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存分布图: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='可视化细粒度注意力权重')

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
    parser.add_argument('--save_dir', type=str, default='./attention_visualization',
                       help='保存结果的目录')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu 或 cuda)')
    parser.add_argument('--num_examples', type=int, default=5,
                       help='可视化的样本数')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='使用哪个数据集split')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("="*80)
    print("🎨 细粒度注意力权重可视化")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}/{args.property}")
    print(f"Save dir: {args.save_dir}")
    print()

    # 1. 加载模型
    print("📥 加载模型...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_config = checkpoint.get('config', None)

    if model_config is None:
        print("❌ Checkpoint中未找到模型配置")
        return

    # 检查是否启用细粒度注意力
    if not getattr(model_config, 'use_fine_grained_attention', False):
        print("❌ 模型未启用细粒度注意力（use_fine_grained_attention=False）")
        print("   请使用启用了细粒度注意力的模型")
        return

    model = ALIGNN(model_config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print("✅ 模型加载成功")
    print(f"✅ 细粒度注意力已启用")

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

    # 3. 提取注意力权重
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA不可用，使用CPU")
        device = 'cpu'

    examples = extract_attention_examples(model, data_loader, device=device,
                                         num_examples=args.num_examples)

    if not examples:
        print("❌ 未能提取注意力权重")
        return

    # 4. 可视化每个样本
    print(f"\n🎨 可视化{len(examples)}个样本的注意力...")
    for idx, example in enumerate(examples):
        print(f"\n  处理样本 {idx+1}/{len(examples)}: {example['material_id']}")

        # 双向注意力
        save_path = os.path.join(args.save_dir,
                                f"attention_sample_{idx+1}_bidirectional.pdf")
        visualize_bidirectional_attention(
            example['atom_to_text'],
            example['text_to_atom'],
            example['atoms'],
            example['tokens'],
            example['material_id'],
            save_path
        )

    # 5. 分析注意力模式
    analyze_attention_patterns(examples, args.save_dir)

    # 6. 可视化注意力分布
    visualize_attention_distribution(examples, args.save_dir)

    print("\n" + "="*80)
    print("✅ 注意力可视化完成！")
    print("="*80)
    print(f"\n生成的文件在: {args.save_dir}")
    print(f"  - attention_sample_*_bidirectional.pdf/png: 双向注意力热图")
    print(f"  - attention_distribution.pdf/png: 注意力权重分布")
    print(f"  - attention_analysis.txt: 注意力模式分析报告")


if __name__ == '__main__':
    main()
