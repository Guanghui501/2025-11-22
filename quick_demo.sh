#!/bin/bash

# 融合机制对比实验快速演示脚本

echo "=================================================="
echo "  融合机制对比实验 - 快速演示"
echo "=================================================="
echo ""

# 检查参数
if [ $# -lt 3 ]; then
    echo "用法: $0 <checkpoint路径> <数据集类型> <属性名称> [数据集根目录]"
    echo ""
    echo "示例:"
    echo "  $0 outputs/best_val_model.pt jarvis formation_energy ./dataset"
    echo ""
    echo "数据集类型: jarvis, mp, class"
    exit 1
fi

CHECKPOINT=$1
DATASET=$2
PROPERTY=$3
ROOT_DIR=${4:-./dataset}

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: Checkpoint文件不存在: $CHECKPOINT"
    exit 1
fi

# 检查数据集目录
if [ ! -d "$ROOT_DIR" ]; then
    echo "❌ 错误: 数据集目录不存在: $ROOT_DIR"
    exit 1
fi

# 创建结果目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="./demo_results_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

echo "配置信息:"
echo "  - Checkpoint: $CHECKPOINT"
echo "  - Dataset: $DATASET/$PROPERTY"
echo "  - Root dir: $ROOT_DIR"
echo "  - Results: $RESULT_DIR"
echo ""

# 检查是否有GPU
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo "✅ 检测到GPU，使用CUDA加速"
else
    DEVICE="cpu"
    echo "⚠️  未检测到GPU，使用CPU（可能较慢）"
fi
echo ""

# ============================================================
# 步骤1: 融合机制对比
# ============================================================

echo "=================================================="
echo "步骤 1/3: 融合机制对比实验"
echo "=================================================="
echo ""
echo "正在运行消融实验，对比不同融合配置的效果..."
echo "这可能需要几分钟时间，请耐心等待..."
echo ""

python compare_fusion_mechanisms.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --root_dir "$ROOT_DIR" \
    --save_dir "$RESULT_DIR/fusion_comparison" \
    --max_samples 300 \
    --device "$DEVICE" \
    --batch_size 16

if [ $? -ne 0 ]; then
    echo "❌ 融合机制对比失败"
    exit 1
fi

echo ""
echo "✅ 融合机制对比完成！"
echo "   结果保存在: $RESULT_DIR/fusion_comparison/"
echo ""

# ============================================================
# 步骤2: 注意力权重可视化
# ============================================================

echo "=================================================="
echo "步骤 2/3: 注意力权重可视化"
echo "=================================================="
echo ""

# 检查模型是否启用细粒度注意力
python -c "
import torch
checkpoint = torch.load('$CHECKPOINT', map_location='cpu', weights_only=False)
config = checkpoint.get('config')
if config and hasattr(config, 'use_fine_grained_attention') and config.use_fine_grained_attention:
    exit(0)
else:
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ 模型启用了细粒度注意力，开始可视化..."
    echo ""

    python visualize_attention_weights.py \
        --checkpoint "$CHECKPOINT" \
        --dataset "$DATASET" \
        --property "$PROPERTY" \
        --root_dir "$ROOT_DIR" \
        --save_dir "$RESULT_DIR/attention_visualization" \
        --num_examples 5 \
        --device "$DEVICE" \
        --batch_size 8

    if [ $? -ne 0 ]; then
        echo "⚠️  注意力可视化失败，但继续下一步"
    else
        echo ""
        echo "✅ 注意力可视化完成！"
        echo "   结果保存在: $RESULT_DIR/attention_visualization/"
        echo ""
    fi
else
    echo "⚠️  模型未启用细粒度注意力，跳过此步骤"
    echo "   要启用注意力可视化，请使用 use_fine_grained_attention=True 训练模型"
    echo ""
fi

# ============================================================
# 步骤3: 生成总结报告
# ============================================================

echo "=================================================="
echo "步骤 3/3: 生成总结报告"
echo "=================================================="
echo ""

SUMMARY_FILE="$RESULT_DIR/SUMMARY.md"

cat > "$SUMMARY_FILE" << 'SUMMARY_EOF'
# 融合机制对比实验总结

## 实验配置

SUMMARY_EOF

echo "- **Checkpoint**: $CHECKPOINT" >> "$SUMMARY_FILE"
echo "- **Dataset**: $DATASET/$PROPERTY" >> "$SUMMARY_FILE"
echo "- **Device**: $DEVICE" >> "$SUMMARY_FILE"
echo "- **Date**: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

cat >> "$SUMMARY_FILE" << 'SUMMARY_EOF'
## 实验结果

### 1. 融合机制对比

查看以下文件了解详细结果:

- **特征空间可视化**: `fusion_comparison/feature_comparison_tsne.pdf`
  - 6个子图展示不同融合阶段的特征空间
  - 观察聚类紧密度和类间分离度

- **定量指标对比**: `fusion_comparison/metrics_comparison.pdf`
  - Silhouette Score (越高越好)
  - Davies-Bouldin Index (越低越好)
  - Separation (越高越好)

- **详细报告**: `fusion_comparison/comparison_report.txt`
  - 数值对比表格
  - 最佳配置分析

### 关键发现

查看 `fusion_comparison/comparison_report.txt` 中的对比表格，重点关注:

1. **Silhouette Score**: 完整融合 vs 基线的提升
2. **Separation**: 类内相似度 - 类间相似度的改善
3. **特征空间**: t-SNE图中聚类的清晰度

SUMMARY_EOF

# 如果有注意力可视化结果
if [ -d "$RESULT_DIR/attention_visualization" ]; then
    cat >> "$SUMMARY_FILE" << 'SUMMARY_EOF'

### 2. 注意力权重分析

查看以下文件了解细粒度注意力:

- **双向注意力热图**: `attention_visualization/attention_sample_*_bidirectional.pdf`
  - 左图: 原子→文本（哪个原子关注哪个词）
  - 右图: 文本→原子（哪个词关注哪个原子）

- **权重分布**: `attention_visualization/attention_distribution.pdf`
  - 注意力权重的统计分布
  - 稀疏度分析

- **模式分析**: `attention_visualization/attention_analysis.txt`
  - Top-5高权重的原子-词对
  - 发现有趣的化学关联

### 注意力模式观察

打开注意力热图，查看:

1. 化学元素原子是否关注对应的元素词？
2. 材料性质词关注哪些原子？
3. 权重分布是否合理（有选择性，不是均匀分布）？

SUMMARY_EOF
fi

cat >> "$SUMMARY_FILE" << 'SUMMARY_EOF'

## 如何解读结果

### 特征空间对比

观察6个子图（如果都有的话）:

1. **Graph (No Fusion)** - 基线图特征
2. **Text (No Fusion)** - 基线文本特征
3. **Graph (+ Middle Fusion)** - 中间融合的影响
4. **Graph (+ Fine-grained Attn)** - 细粒度注意力的影响
5. **Graph (+ Cross-modal Attn)** - 全局注意力的影响
6. **Fused (All Mechanisms)** - 完整融合（应该最好）

**好的结果**:
- ✅ 从基线到完整融合，聚类越来越紧密
- ✅ 不同类别的簇分离越来越明显
- ✅ 决策边界越来越清晰

### 定量指标

**Silhouette Score** (范围-1到1):
- 0.7-1.0: 优秀
- 0.5-0.7: 良好
- 0.25-0.5: 一般
- <0.25: 较差

**期望趋势**:
- 融合机制越多，Silhouette越高
- Davies-Bouldin越低
- Separation越高

### 组合优势的体现

如果实验成功，应该看到:

1. **累积效应**: 每增加一个融合机制，指标都有提升
2. **协同作用**: 组合效果 > 单一机制效果之和
3. **可解释性**: 注意力权重显示合理的原子-词对应

## 后续分析建议

1. **对比表格**: 制作不同配置的指标对比表
2. **消融研究**: 系统地移除每个机制，观察影响
3. **案例分析**: 选择特定样本深入分析注意力模式
4. **论文图片**: 使用生成的PDF作为论文插图

## 文件清单

```
demo_results_YYYYMMDD_HHMMSS/
├── fusion_comparison/
│   ├── feature_comparison_tsne.pdf      # 特征空间对比
│   ├── feature_comparison_tsne.png
│   ├── metrics_comparison.pdf           # 定量指标对比
│   ├── metrics_comparison.png
│   └── comparison_report.txt            # 详细报告
├── attention_visualization/             # (如果有)
│   ├── attention_sample_*_bidirectional.pdf
│   ├── attention_distribution.pdf
│   └── attention_analysis.txt
└── SUMMARY.md                           # 本文件
```

## 相关文档

- `FUSION_COMPARISON_GUIDE.md` - 完整使用指南
- `MODAL_FUSION_MECHANISMS.md` - 融合机制技术文档
- `README_FUSION_MECHANISMS.md` - 快速参考

---

**实验完成时间**: $(date)
SUMMARY_EOF

echo "✅ 总结报告已生成: $SUMMARY_FILE"
echo ""

# ============================================================
# 完成
# ============================================================

echo "=================================================="
echo "  🎉 所有实验完成！"
echo "=================================================="
echo ""
echo "结果保存在: $RESULT_DIR/"
echo ""
echo "快速查看:"
echo "  1. 阅读总结: cat $SUMMARY_FILE"
echo "  2. 查看对比图: open $RESULT_DIR/fusion_comparison/feature_comparison_tsne.pdf"
echo "  3. 查看指标: open $RESULT_DIR/fusion_comparison/metrics_comparison.pdf"

if [ -d "$RESULT_DIR/attention_visualization" ]; then
    echo "  4. 查看注意力: open $RESULT_DIR/attention_visualization/attention_sample_1_bidirectional.pdf"
fi

echo ""
echo "详细使用指南: cat FUSION_COMPARISON_GUIDE.md"
echo ""
echo "如有问题，请查看文档或提issue"
echo "=================================================="
