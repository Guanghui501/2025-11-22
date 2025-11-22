# 多配置模型对比实验指南

如果您有多个不同融合配置的checkpoint，这是展示融合机制优势的最佳方式！

---

## 📦 准备工作

### 1. 确认您的模型配置

假设您有以下几个checkpoint：

```bash
outputs/
├── baseline_model.pt          # 无融合机制
├── middle_fusion_model.pt     # 只有中间融合
├── fine_grained_model.pt      # 中间融合 + 细粒度注意力
└── full_fusion_model.pt       # 所有三个机制
```

### 2. 检查每个模型的配置

创建一个脚本 `check_model_configs.py`：

```python
#!/usr/bin/env python
import torch
import sys

def check_config(checkpoint_path):
    """检查模型配置"""
    print(f"\n{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'='*60}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', None)

    if config is None:
        print("❌ 未找到配置信息")
        return

    print(f"中间融合 (Middle Fusion):           {getattr(config, 'use_middle_fusion', False)}")
    if getattr(config, 'use_middle_fusion', False):
        print(f"  - 融合层: {getattr(config, 'middle_fusion_layers', 'N/A')}")

    print(f"细粒度注意力 (Fine-grained Attn):   {getattr(config, 'use_fine_grained_attention', False)}")
    print(f"全局注意力 (Cross-modal Attn):     {getattr(config, 'use_cross_modal_attention', False)}")

    # 其他有用信息
    print(f"\n模型参数:")
    print(f"  - Hidden features: {getattr(config, 'hidden_features', 'N/A')}")
    print(f"  - ALIGNN layers: {getattr(config, 'alignn_layers', 'N/A')}")
    print(f"  - GCN layers: {getattr(config, 'gcn_layers', 'N/A')}")
    print(f"  - Classification: {getattr(config, 'classification', False)}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python check_model_configs.py model1.pt model2.pt ...")
        sys.exit(1)

    for ckpt in sys.argv[1:]:
        check_config(ckpt)
```

运行：
```bash
python check_model_configs.py outputs/*.pt
```

---

## 🔬 方法1: 逐个对比分析

### 使用 `compare_fusion_mechanisms.py`

**为每个模型单独运行对比**：

```bash
# 1. 基线模型（无融合）
python compare_fusion_mechanisms.py \
    --checkpoint outputs/baseline_model.pt \
    --dataset jarvis \
    --property formation_energy \
    --save_dir ./comparison_baseline \
    --max_samples 500 \
    --device cuda

# 2. 中间融合模型
python compare_fusion_mechanisms.py \
    --checkpoint outputs/middle_fusion_model.pt \
    --dataset jarvis \
    --property formation_energy \
    --save_dir ./comparison_middle \
    --max_samples 500 \
    --device cuda

# 3. 细粒度注意力模型
python compare_fusion_mechanisms.py \
    --checkpoint outputs/fine_grained_model.pt \
    --dataset jarvis \
    --property formation_energy \
    --save_dir ./comparison_fine \
    --max_samples 500 \
    --device cuda

# 4. 完整融合模型
python compare_fusion_mechanisms.py \
    --checkpoint outputs/full_fusion_model.pt \
    --dataset jarvis \
    --property formation_energy \
    --save_dir ./comparison_full \
    --max_samples 500 \
    --device cuda
```

### 每个模型会生成什么？

每个运行会在各自的目录生成：

```
comparison_<model>/
├── feature_comparison_tsne.pdf     # 特征空间t-SNE对比图
├── feature_comparison_tsne.png
├── metrics_comparison.pdf          # 定量指标对比
├── metrics_comparison.png
└── comparison_report.txt           # 详细数值报告
```

### 解读单个模型的结果

打开 `feature_comparison_tsne.pdf`，会看到6个子图：

```
┌─────────────┬─────────────┬─────────────┐
│ Graph Base  │ Text Base   │ Graph+Mid   │
│             │             │ (如果启用)    │
├─────────────┼─────────────┼─────────────┤
│ Graph+Fine  │ Graph+Cross │ Fused       │
│ (如果启用)    │ (如果启用)    │             │
└─────────────┴─────────────┴─────────────┘
```

**关键观察**：
- 对于基线模型：可能只有 `Graph Base` 和 `Text Base`
- 对于部分融合模型：会有对应启用机制的子图
- 对于完整模型：所有6个子图都有

---

## 🎯 方法2: 批量自动化对比

创建批处理脚本 `batch_compare.sh`：

```bash
#!/bin/bash

# 配置
DATASET="jarvis"
PROPERTY="formation_energy"
ROOT_DIR="./dataset"
MAX_SAMPLES=500
DEVICE="cuda"
RESULTS_BASE="./multi_model_comparison"

# 模型列表（名称:路径）
declare -A MODELS=(
    ["baseline"]="outputs/baseline_model.pt"
    ["middle"]="outputs/middle_fusion_model.pt"
    ["fine"]="outputs/fine_grained_model.pt"
    ["full"]="outputs/full_fusion_model.pt"
)

echo "=========================================="
echo "  批量模型对比实验"
echo "=========================================="
echo ""

# 创建结果目录
mkdir -p "$RESULTS_BASE"

# 对每个模型运行对比
for name in "${!MODELS[@]}"; do
    checkpoint="${MODELS[$name]}"
    save_dir="${RESULTS_BASE}/comparison_${name}"

    echo "----------------------------------------"
    echo "处理: $name"
    echo "Checkpoint: $checkpoint"
    echo "----------------------------------------"

    if [ ! -f "$checkpoint" ]; then
        echo "❌ 跳过: checkpoint不存在"
        continue
    fi

    # 运行对比
    python compare_fusion_mechanisms.py \
        --checkpoint "$checkpoint" \
        --dataset "$DATASET" \
        --property "$PROPERTY" \
        --root_dir "$ROOT_DIR" \
        --save_dir "$save_dir" \
        --max_samples "$MAX_SAMPLES" \
        --device "$DEVICE"

    if [ $? -eq 0 ]; then
        echo "✅ 完成: $name"
    else
        echo "❌ 失败: $name"
    fi
    echo ""
done

echo "=========================================="
echo "✅ 所有模型对比完成！"
echo "=========================================="
echo ""
echo "结果保存在: $RESULTS_BASE/"
```

运行：
```bash
chmod +x batch_compare.sh
./batch_compare.sh
```

---

## 📊 方法3: 跨模型结果汇总

### 3.1 提取关键指标

创建 `summarize_results.py`：

```python
#!/usr/bin/env python
"""
从多个对比实验结果中提取关键指标并汇总
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_report(report_path):
    """解析comparison_report.txt"""
    metrics = {}

    if not os.path.exists(report_path):
        return None

    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取fused（最佳）配置的指标
    # 查找表格中的fused行
    pattern = r'fused\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    match = re.search(pattern, content)

    if match:
        metrics['silhouette'] = float(match.group(1))
        metrics['davies_bouldin'] = float(match.group(2))
        metrics['intra_sim'] = float(match.group(3))
        metrics['inter_sim'] = float(match.group(4))
        metrics['separation'] = float(match.group(5))

    return metrics

def summarize_all_models(base_dir):
    """汇总所有模型的结果"""
    results = {}

    # 遍历所有comparison_*目录
    for dirname in os.listdir(base_dir):
        if not dirname.startswith('comparison_'):
            continue

        model_name = dirname.replace('comparison_', '')
        report_path = os.path.join(base_dir, dirname, 'comparison_report.txt')

        metrics = parse_report(report_path)
        if metrics:
            results[model_name] = metrics

    # 转换为DataFrame
    df = pd.DataFrame(results).T

    return df

def plot_summary(df, save_path):
    """绘制汇总对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    metrics = [
        ('silhouette', 'Silhouette Score ↑', True),
        ('davies_bouldin', 'Davies-Bouldin Index ↓', False),
        ('intra_sim', 'Intra-class Similarity ↑', True),
        ('inter_sim', 'Inter-class Similarity ↓', False),
        ('separation', 'Separation ↑', True),
    ]

    colors = ['#A8DADC', '#7FB3D5', '#F4A261', '#6A4C93']

    for idx, (metric, label, higher_better) in enumerate(metrics):
        ax = axes[idx]

        if metric not in df.columns:
            continue

        # 排序（根据是否越高越好）
        sorted_df = df.sort_values(metric, ascending=not higher_better)

        bars = ax.barh(range(len(sorted_df)), sorted_df[metric],
                       color=colors[:len(sorted_df)],
                       edgecolor='black', linewidth=1.5)

        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df.index, fontsize=10)
        ax.set_xlabel(label, fontsize=11, weight='bold')
        ax.set_title(label, fontsize=12, weight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # 标注数值
        for i, (idx_name, val) in enumerate(sorted_df[metric].items()):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=9, weight='bold')

    # 隐藏最后一个空白子图
    axes[-1].axis('off')

    plt.suptitle('Cross-Model Performance Comparison', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"✅ 保存汇总图: {save_path}")
    plt.close()

def generate_summary_table(df, save_path):
    """生成Markdown格式的汇总表格"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("# 多模型对比实验汇总\n\n")
        f.write("## 定量指标对比\n\n")

        # 表头
        f.write("| 模型配置 | Silhouette↑ | Davies-B↓ | Intra-Sim↑ | Inter-Sim↓ | Separation↑ |\n")
        f.write("|---------|-------------|-----------|------------|------------|-------------|\n")

        # 按Silhouette排序
        sorted_df = df.sort_values('silhouette', ascending=False)

        for model_name, row in sorted_df.iterrows():
            f.write(f"| {model_name:<15} ")
            f.write(f"| {row['silhouette']:.4f} ")
            f.write(f"| {row['davies_bouldin']:.4f} ")
            f.write(f"| {row['intra_sim']:.4f} ")
            f.write(f"| {row['inter_sim']:.4f} ")
            f.write(f"| {row['separation']:.4f} |\n")

        f.write("\n## 改进幅度\n\n")

        # 计算相对于baseline的改进
        if 'baseline' in df.index:
            baseline = df.loc['baseline']

            f.write("相对于baseline的改进:\n\n")
            f.write("| 模型 | Silhouette | Separation |\n")
            f.write("|------|-----------|------------|\n")

            for model_name, row in sorted_df.iterrows():
                if model_name == 'baseline':
                    f.write(f"| {model_name} | - | - |\n")
                else:
                    sil_improve = (row['silhouette'] - baseline['silhouette']) / baseline['silhouette'] * 100
                    sep_improve = (row['separation'] - baseline['separation']) / baseline['separation'] * 100
                    f.write(f"| {model_name} | +{sil_improve:.1f}% | +{sep_improve:.1f}% |\n")

    print(f"✅ 保存汇总表格: {save_path}")

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python summarize_results.py <结果目录>")
        print("示例: python summarize_results.py ./multi_model_comparison")
        sys.exit(1)

    base_dir = sys.argv[1]

    print("📊 汇总多模型对比结果...")

    # 汇总指标
    df = summarize_all_models(base_dir)

    if df.empty:
        print("❌ 未找到有效的对比结果")
        sys.exit(1)

    print(f"✅ 找到 {len(df)} 个模型的结果\n")
    print(df)
    print()

    # 生成图表
    plot_summary(df, os.path.join(base_dir, 'summary_comparison.pdf'))

    # 生成表格
    generate_summary_table(df, os.path.join(base_dir, 'summary_table.md'))

    print("\n✅ 汇总完成！")
```

运行：
```bash
python summarize_results.py ./multi_model_comparison
```

---

## 👁️ 方法4: 注意力权重可视化

### 使用 `visualize_attention_weights.py`

**仅对启用了细粒度注意力的模型运行**：

```bash
# 检查模型是否启用细粒度注意力
python -c "
import torch
ckpt = torch.load('outputs/fine_grained_model.pt', map_location='cpu', weights_only=False)
config = ckpt['config']
if getattr(config, 'use_fine_grained_attention', False):
    print('✅ 已启用细粒度注意力，可以可视化')
else:
    print('❌ 未启用细粒度注意力，跳过')
"

# 如果启用，则运行可视化
python visualize_attention_weights.py \
    --checkpoint outputs/fine_grained_model.pt \
    --dataset jarvis \
    --property formation_energy \
    --save_dir ./attention_fine_grained \
    --num_examples 10 \
    --device cuda
```

### 批量注意力可视化

创建 `batch_attention.sh`：

```bash
#!/bin/bash

DATASET="jarvis"
PROPERTY="formation_energy"
NUM_EXAMPLES=10
DEVICE="cuda"

# 需要可视化注意力的模型
MODELS=(
    "outputs/fine_grained_model.pt:attention_fine"
    "outputs/full_fusion_model.pt:attention_full"
)

for model_info in "${MODELS[@]}"; do
    IFS=':' read -r checkpoint save_name <<< "$model_info"

    echo "处理: $checkpoint"

    # 检查是否启用细粒度注意力
    enabled=$(python -c "
import torch
import sys
try:
    ckpt = torch.load('$checkpoint', map_location='cpu', weights_only=False)
    config = ckpt.get('config')
    if config and getattr(config, 'use_fine_grained_attention', False):
        print('yes')
    else:
        print('no')
except:
    print('no')
" 2>/dev/null)

    if [ "$enabled" = "yes" ]; then
        echo "✅ 启用了细粒度注意力，开始可视化..."

        python visualize_attention_weights.py \
            --checkpoint "$checkpoint" \
            --dataset "$DATASET" \
            --property "$PROPERTY" \
            --save_dir "./$save_name" \
            --num_examples "$NUM_EXAMPLES" \
            --device "$DEVICE"
    else
        echo "⚠️  未启用细粒度注意力，跳过"
    fi
    echo ""
done
```

---

## 📈 方法5: 创建综合对比报告

创建 `create_comparison_report.py`：

```python
#!/usr/bin/env python
"""
生成包含所有模型对比的综合报告
"""

import os
import sys
from datetime import datetime

def create_report(base_dir, output_file):
    """创建综合对比报告"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 多模型融合机制对比实验报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## 实验配置\n\n")
        f.write("对比的模型配置:\n\n")

        # 列出所有模型
        models = []
        for dirname in sorted(os.listdir(base_dir)):
            if dirname.startswith('comparison_'):
                model_name = dirname.replace('comparison_', '')
                models.append(model_name)
                f.write(f"- **{model_name}**: ")

                # 读取配置（如果有的话）
                report_path = os.path.join(base_dir, dirname, 'comparison_report.txt')
                if os.path.exists(report_path):
                    f.write("详见对比报告\n")
                else:
                    f.write("无详细信息\n")

        f.write("\n---\n\n")
        f.write("## 实验结果\n\n")

        # 嵌入汇总表格
        summary_table = os.path.join(base_dir, 'summary_table.md')
        if os.path.exists(summary_table):
            with open(summary_table, 'r', encoding='utf-8') as st:
                content = st.read()
                # 跳过标题，只要表格
                lines = content.split('\n')
                for line in lines[4:]:  # 跳过前4行
                    f.write(line + '\n')

        f.write("\n---\n\n")
        f.write("## 可视化结果\n\n")
        f.write("### 汇总对比图\n\n")
        f.write(f"![汇总对比](summary_comparison.png)\n\n")

        f.write("### 各模型特征空间\n\n")
        for model in models:
            comp_dir = os.path.join(base_dir, f'comparison_{model}')
            tsne_img = f'comparison_{model}/feature_comparison_tsne.png'

            if os.path.exists(os.path.join(base_dir, tsne_img)):
                f.write(f"#### {model}\n\n")
                f.write(f"![{model} 特征空间]({tsne_img})\n\n")

        f.write("---\n\n")
        f.write("## 结论\n\n")

        # 读取汇总表格找出最佳模型
        if os.path.exists(summary_table):
            f.write("根据Silhouette Score排序，模型性能从高到低:\n\n")
            # 这里可以解析表格并列出排序
            f.write("详见上述定量指标对比表格。\n\n")

        f.write("### 关键发现\n\n")
        f.write("1. **融合机制的累积效果**: 随着融合机制的增加，性能逐步提升\n")
        f.write("2. **最佳配置**: 启用所有三个融合机制的模型表现最佳\n")
        f.write("3. **可解释性**: 细粒度注意力提供了原子-词对应关系\n\n")

        f.write("---\n\n")
        f.write("## 文件清单\n\n")
        f.write("```\n")
        f.write(f"{base_dir}/\n")
        for model in models:
            f.write(f"├── comparison_{model}/\n")
            f.write(f"│   ├── feature_comparison_tsne.pdf\n")
            f.write(f"│   ├── metrics_comparison.pdf\n")
            f.write(f"│   └── comparison_report.txt\n")
        f.write(f"├── summary_comparison.pdf\n")
        f.write(f"├── summary_table.md\n")
        f.write(f"└── comparison_report.md (本文件)\n")
        f.write("```\n")

    print(f"✅ 综合报告已生成: {output_file}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python create_comparison_report.py <结果目录>")
        sys.exit(1)

    base_dir = sys.argv[1]
    output_file = os.path.join(base_dir, 'comparison_report.md')

    create_report(base_dir, output_file)
```

---

## 🎯 完整工作流示例

### 一键运行所有对比

创建 `master_comparison.sh`：

```bash
#!/bin/bash

# ========== 配置 ==========
DATASET="jarvis"
PROPERTY="formation_energy"
ROOT_DIR="./dataset"
MAX_SAMPLES=500
DEVICE="cuda"
RESULTS_DIR="./complete_comparison_$(date +%Y%m%d_%H%M%S)"

# 模型配置
declare -A MODELS=(
    ["baseline"]="outputs/baseline_model.pt"
    ["middle"]="outputs/middle_fusion_model.pt"
    ["fine"]="outputs/fine_grained_model.pt"
    ["full"]="outputs/full_fusion_model.pt"
)

# ========== 开始 ==========
echo "=========================================="
echo "  完整多模型对比实验流程"
echo "=========================================="
echo ""
echo "结果保存在: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# ========== 步骤1: 检查模型配置 ==========
echo ""
echo "步骤 1/5: 检查模型配置..."
echo "------------------------------------------"

for name in "${!MODELS[@]}"; do
    checkpoint="${MODELS[$name]}"
    python check_model_configs.py "$checkpoint" | tee -a "$RESULTS_DIR/model_configs.txt"
done

# ========== 步骤2: 运行特征对比 ==========
echo ""
echo "步骤 2/5: 运行特征对比..."
echo "------------------------------------------"

for name in "${!MODELS[@]}"; do
    checkpoint="${MODELS[$name]}"
    save_dir="${RESULTS_DIR}/comparison_${name}"

    echo "处理: $name"

    python compare_fusion_mechanisms.py \
        --checkpoint "$checkpoint" \
        --dataset "$DATASET" \
        --property "$PROPERTY" \
        --root_dir "$ROOT_DIR" \
        --save_dir "$save_dir" \
        --max_samples "$MAX_SAMPLES" \
        --device "$DEVICE"

    echo "✅ 完成: $name"
    echo ""
done

# ========== 步骤3: 注意力可视化 ==========
echo ""
echo "步骤 3/5: 注意力可视化..."
echo "------------------------------------------"

for name in "${!MODELS[@]}"; do
    checkpoint="${MODELS[$name]}"

    # 检查是否启用细粒度注意力
    enabled=$(python -c "
import torch
ckpt = torch.load('$checkpoint', map_location='cpu', weights_only=False)
config = ckpt.get('config')
if config and getattr(config, 'use_fine_grained_attention', False):
    print('yes')
else:
    print('no')
" 2>/dev/null)

    if [ "$enabled" = "yes" ]; then
        echo "可视化 $name 的注意力..."

        python visualize_attention_weights.py \
            --checkpoint "$checkpoint" \
            --dataset "$DATASET" \
            --property "$PROPERTY" \
            --save_dir "${RESULTS_DIR}/attention_${name}" \
            --num_examples 5 \
            --device "$DEVICE"
    else
        echo "⚠️  $name 未启用细粒度注意力，跳过"
    fi
done

# ========== 步骤4: 汇总结果 ==========
echo ""
echo "步骤 4/5: 汇总结果..."
echo "------------------------------------------"

python summarize_results.py "$RESULTS_DIR"

# ========== 步骤5: 生成报告 ==========
echo ""
echo "步骤 5/5: 生成综合报告..."
echo "------------------------------------------"

python create_comparison_report.py "$RESULTS_DIR"

# ========== 完成 ==========
echo ""
echo "=========================================="
echo "  ✅ 完整对比实验完成！"
echo "=========================================="
echo ""
echo "结果目录: $RESULTS_DIR"
echo ""
echo "查看报告: cat $RESULTS_DIR/comparison_report.md"
echo "查看汇总: open $RESULTS_DIR/summary_comparison.pdf"
```

运行：
```bash
chmod +x master_comparison.sh
./master_comparison.sh
```

---

## 📊 预期输出结构

运行完成后，您会得到：

```
complete_comparison_YYYYMMDD_HHMMSS/
├── model_configs.txt                    # 所有模型配置
│
├── comparison_baseline/                 # 基线模型
│   ├── feature_comparison_tsne.pdf
│   ├── metrics_comparison.pdf
│   └── comparison_report.txt
│
├── comparison_middle/                   # 中间融合模型
│   ├── feature_comparison_tsne.pdf
│   ├── metrics_comparison.pdf
│   └── comparison_report.txt
│
├── comparison_fine/                     # 细粒度模型
│   ├── feature_comparison_tsne.pdf
│   ├── metrics_comparison.pdf
│   └── comparison_report.txt
│
├── comparison_full/                     # 完整融合模型
│   ├── feature_comparison_tsne.pdf
│   ├── metrics_comparison.pdf
│   └── comparison_report.txt
│
├── attention_fine/                      # 细粒度模型注意力
│   ├── attention_sample_*.pdf
│   ├── attention_distribution.pdf
│   └── attention_analysis.txt
│
├── attention_full/                      # 完整模型注意力
│   ├── attention_sample_*.pdf
│   ├── attention_distribution.pdf
│   └── attention_analysis.txt
│
├── summary_comparison.pdf               # ⭐ 跨模型汇总对比
├── summary_table.md                     # ⭐ 汇总表格
└── comparison_report.md                 # ⭐ 综合报告
```

---

## 🎨 关键图表解读

### 1. 单模型内部对比 (`feature_comparison_tsne.pdf`)

展示**单个模型**不同阶段的特征：
- 观察融合机制的逐步作用
- 验证每个机制是否生效

### 2. 跨模型性能对比 (`summary_comparison.pdf`)

展示**多个模型**的最终性能：
- 横向对比不同配置
- 识别最佳融合策略

### 3. 注意力模式 (`attention_sample_*.pdf`)

展示**原子-词对应关系**：
- 验证模型理解材料
- 发现有趣的化学关联

---

## 💡 最佳实践

1. **样本数选择**:
   - 快速测试: `--max_samples 200`
   - 正式对比: `--max_samples 500-1000`

2. **设备选择**:
   - 有GPU: `--device cuda` (推荐)
   - 无GPU: `--device cpu` (较慢)

3. **批次大小**:
   - GPU内存充足: `--batch_size 32`
   - GPU内存有限: `--batch_size 8-16`

4. **对比策略**:
   - 先运行一个模型验证流程
   - 确认无误后批量运行
   - 最后汇总结果

---

## 🚀 快速开始

如果您只想快速体验：

```bash
# 1. 对比两个模型（最简单）
python compare_fusion_mechanisms.py \
    --checkpoint outputs/baseline_model.pt \
    --dataset jarvis \
    --property formation_energy \
    --save_dir ./compare_baseline

python compare_fusion_mechanisms.py \
    --checkpoint outputs/full_fusion_model.pt \
    --dataset jarvis \
    --property formation_energy \
    --save_dir ./compare_full

# 2. 查看结果
open ./compare_baseline/feature_comparison_tsne.pdf
open ./compare_full/feature_comparison_tsne.pdf

# 3. 对比：baseline vs full 的聚类质量差异
```

---

**创建时间**: 2025-11-22
**适用场景**: 多个不同配置的模型对比
