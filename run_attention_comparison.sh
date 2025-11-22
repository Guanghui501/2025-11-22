#!/bin/bash
# 运行注意力权重对比分析

echo "=========================================="
echo "注意力权重对比分析"
echo "=========================================="
echo ""
echo "对比全模态（有中期融合）vs 无中期融合模型"
echo ""

# 配置
CHECKPOINT_FULL="output_100epochs_42_bs128_sw_ju/mbj_bandgap/best_val_model.pt"
CHECKPOINT_NO_MIDDLE="output_100epochs_42_bs128_sw_ju_houqi/mbj_bandgap/best_val_model.pt"
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset/jarvis/mbj_bandgap"
DATASET="dft_3d"
PROPERTY="mbj_bandgap"
SAVE_DIR="./attention_analysis"
NUM_SAMPLES=100
BATCH_SIZE=64

echo "配置信息:"
echo "  全模态模型: $CHECKPOINT_FULL"
echo "  无中期模型: $CHECKPOINT_NO_MIDDLE"
echo "  数据目录: $ROOT_DIR"
echo "  数据集: $DATASET"
echo "  属性: $PROPERTY"
echo "  样本数: $NUM_SAMPLES"
echo "  批大小: $BATCH_SIZE"
echo ""

# 检查模型文件是否存在
if [ ! -f "$CHECKPOINT_FULL" ]; then
    echo "❌ 错误: 找不到全模态模型文件: $CHECKPOINT_FULL"
    exit 1
fi

if [ ! -f "$CHECKPOINT_NO_MIDDLE" ]; then
    echo "❌ 错误: 找不到无中期模型文件: $CHECKPOINT_NO_MIDDLE"
    exit 1
fi

# 检查数据目录
if [ ! -d "$ROOT_DIR" ]; then
    echo "❌ 错误: 找不到数据目录: $ROOT_DIR"
    exit 1
fi

echo "✅ 文件检查通过，开始分析..."
echo ""

# 运行分析
python compare_attention_weights.py \
    --checkpoint_full "$CHECKPOINT_FULL" \
    --checkpoint_no_middle "$CHECKPOINT_NO_MIDDLE" \
    --root_dir "$ROOT_DIR" \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --save_dir "$SAVE_DIR" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 分析完成！"
    echo "=========================================="
    echo ""
    echo "结果保存在: $SAVE_DIR"
    echo ""
    echo "生成的文件:"
    echo "  - attention_statistics_comparison.png"
    echo "  - attention_statistics.csv"
    echo "  - attention_heatmap_example_*.png"
    echo ""
    echo "详细使用指南请查看: ATTENTION_ANALYSIS_GUIDE.md"
else
    echo ""
    echo "❌ 分析失败，请检查错误信息"
    exit 1
fi
