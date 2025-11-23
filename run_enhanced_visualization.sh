#!/bin/bash
# 运行增强版注意力热图可视化

echo "=========================================="
echo "增强版注意力热图可视化"
echo "=========================================="
echo ""
echo "显示实际的token文本和详细统计信息"
echo ""

CHECKPOINT_FULL="output_100epochs_42_bs128_sw_ju/mbj_bandgap/best_val_model.pt"
CHECKPOINT_NO_MIDDLE="output_100epochs_42_bs128_sw_ju_houqi/mbj_bandgap/best_val_model.pt"
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset/jarvis/mbj_bandgap"
DATASET="dft_3d"
PROPERTY="mbj_bandgap"
SAVE_DIR="./attention_enhanced"
NUM_EXAMPLES=3

python visualize_attention_enhanced.py \
    --checkpoint_full "$CHECKPOINT_FULL" \
    --checkpoint_no_middle "$CHECKPOINT_NO_MIDDLE" \
    --root_dir "$ROOT_DIR" \
    --dataset "$DATASET" \
    --property "$PROPERTY" \
    --save_dir "$SAVE_DIR" \
    --num_examples "$NUM_EXAMPLES"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 成功！查看结果："
    echo "   ls -lh $SAVE_DIR"
    echo ""
    echo "生成的文件："
    echo "  - enhanced_heatmap_*.png  : 带有token文本标签的热图"
    echo "  - detailed_analysis_*.png : 包含统计信息的详细分析图"
else
    echo "❌ 生成失败"
fi
