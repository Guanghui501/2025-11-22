#!/bin/bash

# ============================================================================
# 多种子消融实验监控脚本（串行执行版本 + Full Model）
# 检查4个实验 × 3个种子 + Full Model × 3个种子 = 15个训练任务的状态
# ============================================================================

BASE_OUTPUT_DIR="./ablation_multi_seed"
FULL_MODEL_DIR="./full_model_multi_seed"

echo "============================================================================"
echo "📊 消融实验状态检查（包括Full Model）"
echo "============================================================================"
echo ""
echo "时间: $(date)"
echo ""

# ============================================================================
# 1. 各实验详细进度
# ============================================================================
echo "============================================================================"
echo "1️⃣  实验详细进度"
echo "============================================================================"
echo ""

# 定义实验配置
declare -A exp_names=(
    [1]="Exp-1: Baseline"
    [2]="Exp-2: +Late Fusion"
    [3]="Exp-3: +Middle Fusion"
    [4]="Exp-4: +Fine-Grained"
    [5]="Exp-5: Full Model"
)

seeds=(42 123 7)

# 检查Exp-1到Exp-4（消融实验）
for exp_num in {1..4}; do
    echo "----------------------------------------"
    echo "${exp_names[$exp_num]}"
    echo "----------------------------------------"

    for seed in "${seeds[@]}"; do
        exp_dir="$BASE_OUTPUT_DIR/exp${exp_num}_seed${seed}"

        if [ -d "$exp_dir" ]; then
            # 检查训练历史文件
            if [ -f "$exp_dir/history_val.json" ]; then
                # 使用Python获取当前轮数和最佳性能
                epoch_info=$(python3 -c "
import json
import sys
try:
    with open('$exp_dir/history_val.json', 'r') as f:
        data = json.load(f)
    epochs = len(data.get('loss', []))

    # 检测任务类型
    if 'mae' in data:
        metric = 'mae'
        best_val = min(data[metric])
        last_val = data[metric][-1]
    elif 'accuracy' in data:
        metric = 'accuracy'
        best_val = max(data[metric])
        last_val = data[metric][-1]
    else:
        metric = 'unknown'
        best_val = 0
        last_val = 0

    print(f'{epochs}|{metric}|{best_val:.4f}|{last_val:.4f}')
except:
    print('0|unknown|0|0')
" 2>/dev/null)

                IFS='|' read -r epochs metric best_val last_val <<< "$epoch_info"

                if [ "$epochs" != "0" ]; then
                    echo "  ✅ Seed $seed: 已完成 $epochs 轮"
                    echo "     最佳 $metric: $best_val | 最后 $metric: $last_val"
                else
                    echo "  🔄 Seed $seed: 进行中..."
                fi
            else
                # 检查training.log是否有内容
                if [ -f "$exp_dir/training.log" ]; then
                    log_size=$(du -h "$exp_dir/training.log" | cut -f1)
                    echo "  🔄 Seed $seed: 进行中... (日志大小: $log_size)"
                else
                    echo "  ⏳ Seed $seed: 准备启动..."
                fi
            fi
        else
            echo "  ⏸️  Seed $seed: 未开始"
        fi
    done

    echo ""
done

# 检查Exp-5（Full Model）
echo "----------------------------------------"
echo "${exp_names[5]}"
echo "----------------------------------------"

for seed in "${seeds[@]}"; do
    # 首先尝试从消融实验目录查找exp5
    exp_dir="$BASE_OUTPUT_DIR/exp5_seed${seed}"

    # 如果没有找到，再尝试从Full Model独立目录查找
    if [ ! -d "$exp_dir" ]; then
        exp_dir="$FULL_MODEL_DIR/full_model_seed${seed}"
    fi

    if [ -d "$exp_dir" ]; then
        # 检查训练历史文件
        if [ -f "$exp_dir/history_val.json" ]; then
            # 使用Python获取当前轮数和最佳性能
            epoch_info=$(python3 -c "
import json
import sys
try:
    with open('$exp_dir/history_val.json', 'r') as f:
        data = json.load(f)
    epochs = len(data.get('loss', []))

    # 检测任务类型
    if 'mae' in data:
        metric = 'mae'
        best_val = min(data[metric])
        last_val = data[metric][-1]
    elif 'accuracy' in data:
        metric = 'accuracy'
        best_val = max(data[metric])
        last_val = data[metric][-1]
    else:
        metric = 'unknown'
        best_val = 0
        last_val = 0

    print(f'{epochs}|{metric}|{best_val:.4f}|{last_val:.4f}')
except:
    print('0|unknown|0|0')
" 2>/dev/null)

            IFS='|' read -r epochs metric best_val last_val <<< "$epoch_info"

            if [ "$epochs" != "0" ]; then
                echo "  ✅ Seed $seed: 已完成 $epochs 轮"
                echo "     最佳 $metric: $best_val | 最后 $metric: $last_val"
            else
                echo "  🔄 Seed $seed: 进行中..."
            fi
        else
            # 检查training.log或nohup.log
            if [ -f "$exp_dir/training.log" ]; then
                log_size=$(du -h "$exp_dir/training.log" | cut -f1)
                echo "  🔄 Seed $seed: 进行中... (日志大小: $log_size)"
            elif [ -f "$exp_dir/nohup.log" ]; then
                log_size=$(du -h "$exp_dir/nohup.log" | cut -f1)
                echo "  🔄 Seed $seed: 进行中... (日志大小: $log_size)"
            else
                echo "  ⏳ Seed $seed: 准备启动..."
            fi
        fi
    else
        echo "  ⏸️  Seed $seed: 未开始"
    fi
done

echo ""

# ============================================================================
# 2. 最新日志摘要
# ============================================================================
echo "============================================================================"
echo "2️⃣  最新日志摘要（各实验最后5行）"
echo "============================================================================"
echo ""

for exp_num in {1..4}; do
    echo "----------------------------------------"
    echo "${exp_names[$exp_num]}"
    echo "----------------------------------------"

    for seed in "${seeds[@]}"; do
        log_file="$BASE_OUTPUT_DIR/exp${exp_num}_seed${seed}/training.log"

        if [ -f "$log_file" ] && [ -s "$log_file" ]; then
            echo ""
            echo "  📝 Seed $seed (最后5行):"
            tail -5 "$log_file" | sed 's/^/     /'
        fi
    done

    echo ""
done

# Full Model日志
echo "----------------------------------------"
echo "${exp_names[5]}"
echo "----------------------------------------"

for seed in "${seeds[@]}"; do
    # 首先尝试从消融实验目录查找training.log
    log_file="$BASE_OUTPUT_DIR/exp5_seed${seed}/training.log"

    # 如果没有找到，尝试从Full Model独立目录查找nohup.log
    if [ ! -f "$log_file" ] || [ ! -s "$log_file" ]; then
        log_file="$FULL_MODEL_DIR/full_model_seed${seed}/nohup.log"
    fi

    if [ -f "$log_file" ] && [ -s "$log_file" ]; then
        echo ""
        echo "  📝 Seed $seed (最后5行):"
        tail -5 "$log_file" | sed 's/^/     /'
    fi
done

echo ""

# ============================================================================
# 3. 结果汇总表
# ============================================================================
echo "============================================================================"
echo "3️⃣  结果汇总表"
echo "============================================================================"
echo ""

# 表头
printf "%-25s | %-12s | %-12s | %-12s\n" "实验配置" "Seed 42" "Seed 123" "Seed 7"
echo "--------------------------------------------------------------------------------"

for exp_num in {1..4}; do
    exp_name="${exp_names[$exp_num]}"

    # 缩短实验名称以适应表格
    case $exp_num in
        1) short_name="Baseline" ;;
        2) short_name="+Late Fusion" ;;
        3) short_name="+Middle Fusion" ;;
        4) short_name="+Fine-Grained" ;;
    esac

    results=()
    for seed in "${seeds[@]}"; do
        exp_dir="$BASE_OUTPUT_DIR/exp${exp_num}_seed${seed}"

        if [ -f "$exp_dir/history_val.json" ]; then
            result=$(python3 -c "
import json
try:
    with open('$exp_dir/history_val.json', 'r') as f:
        data = json.load(f)

    if 'mae' in data:
        metric = 'MAE'
        best_val = min(data['mae'])
    elif 'accuracy' in data:
        metric = 'Acc'
        best_val = max(data['accuracy'])
    else:
        metric = '?'
        best_val = 0

    print(f'{metric}:{best_val:.4f}')
except:
    print('N/A')
" 2>/dev/null)
            results+=("$result")
        else
            results+=("Running...")
        fi
    done

    printf "%-25s | %-12s | %-12s | %-12s\n" \
        "$short_name" \
        "${results[0]}" \
        "${results[1]}" \
        "${results[2]}"
done

# Full Model结果
short_name="Full Model"
results=()
for seed in "${seeds[@]}"; do
    # 首先尝试从消融实验目录查找
    exp_dir="$BASE_OUTPUT_DIR/exp5_seed${seed}"

    # 如果没有找到，尝试从Full Model独立目录查找
    if [ ! -f "$exp_dir/history_val.json" ]; then
        exp_dir="$FULL_MODEL_DIR/full_model_seed${seed}"
    fi

    if [ -f "$exp_dir/history_val.json" ]; then
        result=$(python3 -c "
import json
try:
    with open('$exp_dir/history_val.json', 'r') as f:
        data = json.load(f)

    if 'mae' in data:
        metric = 'MAE'
        best_val = min(data['mae'])
    elif 'accuracy' in data:
        metric = 'Acc'
        best_val = max(data['accuracy'])
    else:
        metric = '?'
        best_val = 0

    print(f'{metric}:{best_val:.4f}')
except:
    print('N/A')
" 2>/dev/null)
        results+=("$result")
    else
        results+=("Running...")
    fi
done

printf "%-25s | %-12s | %-12s | %-12s\n" \
    "$short_name" \
    "${results[0]}" \
    "${results[1]}" \
    "${results[2]}"

echo ""

# ============================================================================
# 4. 磁盘使用情况
# ============================================================================
echo "============================================================================"
echo "4️⃣  磁盘使用情况"
echo "============================================================================"
echo ""

# 消融实验磁盘使用（包括Full Model）
if [ -d "$BASE_OUTPUT_DIR" ]; then
    ablation_size=$(du -sh "$BASE_OUTPUT_DIR" | cut -f1)
    echo "  消融实验总大小（包括Full Model）: $ablation_size"
    echo ""
    echo "  各实验大小:"

    for exp_num in {1..5}; do
        exp_total=0
        for seed in "${seeds[@]}"; do
            exp_dir="$BASE_OUTPUT_DIR/exp${exp_num}_seed${seed}"
            if [ -d "$exp_dir" ]; then
                size=$(du -sm "$exp_dir" | cut -f1)
                exp_total=$((exp_total + size))
            fi
        done

        if [ $exp_total -gt 0 ]; then
            echo "    ${exp_names[$exp_num]}: ${exp_total}MB"
        fi
    done
    echo ""
fi

# Full Model独立目录磁盘使用（如果存在）
if [ -d "$FULL_MODEL_DIR" ]; then
    full_model_size=$(du -sh "$FULL_MODEL_DIR" | cut -f1)
    echo "  Full Model独立目录总大小: $full_model_size"
    echo "  （注：如果Full Model在上面已统计，此处为独立后台运行的Full Model）"
    echo ""
    echo "  各Full Model训练大小:"

    full_total=0
    for seed in "${seeds[@]}"; do
        model_dir="$FULL_MODEL_DIR/full_model_seed${seed}"
        if [ -d "$model_dir" ]; then
            size=$(du -sm "$model_dir" | cut -f1)
            full_total=$((full_total + size))
            echo "    Seed $seed: ${size}MB"
        fi
    done

    if [ $full_total -gt 0 ]; then
        echo ""
        echo "    Total: ${full_total}MB"
    fi
    echo ""
fi

# ============================================================================
# 5. 快捷监控命令
# ============================================================================
echo "============================================================================"
echo "📝 快捷监控命令"
echo "============================================================================"
echo ""
echo "  查看消融实验日志 (例如 Exp1, Seed42):"
echo "    tail -f $BASE_OUTPUT_DIR/exp1_seed42/training.log"
echo ""
echo "  查看Full Model日志 (例如 Seed42):"
echo "    tail -f $FULL_MODEL_DIR/full_model_seed42/nohup.log"
echo ""
echo "  查看消融实验主日志:"
echo "    tail -f $BASE_OUTPUT_DIR/launch_log_*.txt"
echo ""
echo "  查看GPU使用:"
echo "    nvidia-smi"
echo ""
echo "  实时监控此脚本:"
echo "    watch -n 60 ./check_ablation_multi_seed_progress.sh"
echo ""
echo "  查看当前正在训练的实验 (查找python进程):"
echo "    ps aux | grep train_with_cross_modal_attention.py"
echo ""
echo "  生成完整结果汇总（包括Full Model）:"
echo "    python summarize_multi_seed_results.py --ablation_dir $BASE_OUTPUT_DIR"
echo ""
echo "============================================================================"
echo ""
