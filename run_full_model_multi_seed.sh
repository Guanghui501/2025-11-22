#!/bin/bash

# ============================================================================
# 全模块（Full Model）多种子训练脚本 - 串行执行
# 运行3个随机种子的完整模型训练，任务一个接一个执行
# ============================================================================


export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=1

# 基础配置
ROOT_DIR="/public/home/ghzhang/crysmmnet-main/dataset"
DATASET="jarvis"
PROPERTY="shear_modulus_gv"
BASE_OUTPUT_DIR="./full_model_multi_seed_shear"

# 训练超参数（与用户提供的完全一致）
EPOCHS=100
BATCH_SIZE=128
LEARNING_RATE=1e-3
WEIGHT_DECAY=5e-4
WARMUP_STEPS=2000
ALIGNN_LAYERS=4
GCN_LAYERS=4
HIDDEN_FEATURES=256
GRAPH_DROPOUT=0.15
CROSS_MODAL_NUM_HEADS=4
MIDDLE_FUSION_LAYERS=2
FINE_GRAINED_HIDDEN_DIM=256
FINE_GRAINED_NUM_HEADS=8
FINE_GRAINED_DROPOUT=0.2
FINE_GRAINED_USE_PROJECTION=True
EARLY_STOPPING_PATIENCE=150
NUM_WORKERS=24

# 随机种子列表
SEEDS=(42 123 7)

# 公共参数
COMMON_ARGS="
    --root_dir $ROOT_DIR \
    --dataset $DATASET \
    --property $PROPERTY \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --warmup_steps $WARMUP_STEPS \
    --alignn_layers $ALIGNN_LAYERS \
    --gcn_layers $GCN_LAYERS \
    --hidden_features $HIDDEN_FEATURES \
    --graph_dropout $GRAPH_DROPOUT \
    --cross_modal_num_heads $CROSS_MODAL_NUM_HEADS \
    --middle_fusion_layers $MIDDLE_FUSION_LAYERS \
    --fine_grained_hidden_dim $FINE_GRAINED_HIDDEN_DIM \
    --fine_grained_num_heads $FINE_GRAINED_NUM_HEADS \
    --fine_grained_dropout $FINE_GRAINED_DROPOUT \
    --fine_grained_use_projection $FINE_GRAINED_USE_PROJECTION \
    --early_stopping_patience $EARLY_STOPPING_PATIENCE \
    --num_workers $NUM_WORKERS
"

# 创建基础输出目录
mkdir -p "$BASE_OUTPUT_DIR"

# 主日志文件
MAIN_LOG="$BASE_OUTPUT_DIR/launch_log_$(date +%Y%m%d_%H%M%S).txt"

# 用于统计完成任务
COMPLETED_COUNT=0
FAILED_COUNT=0

echo "============================================================================" | tee -a "$MAIN_LOG"
echo "🚀 启动全模块训练（多种子版本 - 串行执行）" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "时间: $(date)" | tee -a "$MAIN_LOG"
echo "数据集: $DATASET/$PROPERTY" | tee -a "$MAIN_LOG"
echo "实验配置: Full Model × 3个种子 = 3个训练任务" | tee -a "$MAIN_LOG"
echo "执行模式: 串行（一个接一个）" | tee -a "$MAIN_LOG"
echo "随机种子: ${SEEDS[@]}" | tee -a "$MAIN_LOG"
echo "基础输出目录: $BASE_OUTPUT_DIR" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# ============================================================================
# 串行运行所有Full Model训练任务
# ============================================================================

for seed in "${SEEDS[@]}"; do
    output_dir="$BASE_OUTPUT_DIR/full_model_seed${seed}"
    log_file="$output_dir/training.log"

    # 创建输出目录
    mkdir -p "$output_dir"

    echo "============================================================================" | tee -a "$MAIN_LOG"
    echo "[$((COMPLETED_COUNT + FAILED_COUNT + 1))/3] 运行: Full Model (seed=$seed)" | tee -a "$MAIN_LOG"
    echo "============================================================================" | tee -a "$MAIN_LOG"
    echo "  开始时间: $(date)" | tee -a "$MAIN_LOG"
    echo "  输出目录: $output_dir" | tee -a "$MAIN_LOG"
    echo "  配置: 所有模块启用 (cross_modal=True, middle_fusion=True, fine_grained=True)" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"

    # 同步执行训练（等待完成）
    python train_with_cross_modal_attention.py \
        $COMMON_ARGS \
        --random_seed $seed \
        --use_cross_modal True \
        --use_middle_fusion True \
        --use_fine_grained_attention True \
        --output_dir "$output_dir" \
        2>&1 | tee "$log_file"

    # 检查退出状态
    exit_code=${PIPESTATUS[0]}

    echo "" | tee -a "$MAIN_LOG"
    if [ $exit_code -eq 0 ]; then
        echo "✅ 完成: Full Model (seed=$seed)" | tee -a "$MAIN_LOG"
        COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
    else
        echo "❌ 失败: Full Model (seed=$seed) - 退出码: $exit_code" | tee -a "$MAIN_LOG"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi

    echo "  结束时间: $(date)" | tee -a "$MAIN_LOG"
    echo "  已完成: $COMPLETED_COUNT, 失败: $FAILED_COUNT, 剩余: $((3 - COMPLETED_COUNT - FAILED_COUNT))" | tee -a "$MAIN_LOG"
    echo "" | tee -a "$MAIN_LOG"
done

# ============================================================================
# 所有训练完成汇总
# ============================================================================
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "🎉 所有Full Model训练执行完成！" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "结束时间: $(date)" | tee -a "$MAIN_LOG"
echo "总计任务: 3" | tee -a "$MAIN_LOG"
echo "成功完成: $COMPLETED_COUNT" | tee -a "$MAIN_LOG"
echo "执行失败: $FAILED_COUNT" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# 生成结果汇总
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "📊 生成结果汇总..." | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

python summarize_full_model_results.py --model_dir "$BASE_OUTPUT_DIR" | tee -a "$MAIN_LOG"

echo "" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "✅ Full Model训练全部完成！" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "查看结果:" | tee -a "$MAIN_LOG"
echo "  - 主日志: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "  - 简明汇总: $BASE_OUTPUT_DIR/full_model_summary.csv" | tee -a "$MAIN_LOG"
echo "  - 详细结果: $BASE_OUTPUT_DIR/full_model_detailed.csv" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "各训练日志:" | tee -a "$MAIN_LOG"
for seed in "${SEEDS[@]}"; do
    log_file="$BASE_OUTPUT_DIR/full_model_seed${seed}/training.log"
    if [ -f "$log_file" ]; then
        echo "  - full_model_seed${seed}: $log_file" | tee -a "$MAIN_LOG"
    fi
done
echo "" | tee -a "$MAIN_LOG"
echo "============================================================================" | tee -a "$MAIN_LOG"
