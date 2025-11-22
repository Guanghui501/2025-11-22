#!/usr/bin/env python
"""
多种子消融实验结果汇总脚本
生成包含均值和标准差的详细CSV报告
"""

import json
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def load_experiment_results(base_dir, exp_num, seed):
    """加载单个实验的结果"""
    exp_dir = Path(base_dir) / f"exp{exp_num}_seed{seed}"

    if not exp_dir.exists():
        return None

    history_val_file = exp_dir / "history_val.json"
    history_train_file = exp_dir / "history_train.json"

    if not history_val_file.exists():
        return None

    try:
        with open(history_val_file, 'r') as f:
            val_history = json.load(f)

        with open(history_train_file, 'r') as f:
            train_history = json.load(f)

        # 检测任务类型
        if 'mae' in val_history:
            task_type = 'regression'
            metric_name = 'mae'
            val_metrics = val_history['mae']
            best_val = min(val_metrics)
            best_epoch = val_metrics.index(best_val)
        elif 'accuracy' in val_history:
            task_type = 'classification'
            metric_name = 'accuracy'
            val_metrics = val_history['accuracy']
            best_val = max(val_metrics)
            best_epoch = val_metrics.index(best_val)
        else:
            return None

        # 提取关键指标
        result = {
            'task_type': task_type,
            'metric_name': metric_name,
            'total_epochs': len(val_history['epochs']),
            'best_epoch': val_history['epochs'][best_epoch],
            'best_val': best_val,
            'final_val': val_metrics[-1],
            'best_train_loss': train_history['loss'][best_epoch],
            'final_train_loss': train_history['loss'][-1],
        }

        # 添加额外指标（如果存在）
        if task_type == 'regression':
            if 'rmse' in val_history:
                result['best_val_rmse'] = min(val_history['rmse'])
                result['final_val_rmse'] = val_history['rmse'][-1]
        elif task_type == 'classification':
            if 'precision' in val_history:
                result['best_val_precision'] = max(val_history['precision'])
                result['final_val_precision'] = val_history['precision'][-1]
            if 'recall' in val_history:
                result['best_val_recall'] = max(val_history['recall'])
                result['final_val_recall'] = val_history['recall'][-1]
            if 'f1' in val_history:
                result['best_val_f1'] = max(val_history['f1'])
                result['final_val_f1'] = val_history['f1'][-1]

        return result

    except Exception as e:
        print(f"警告: 读取 {exp_dir} 时出错: {e}")
        return None


def load_full_model_results(model_dir):
    """加载Full Model的结果"""
    model_dir = Path(model_dir)

    if not model_dir.exists():
        return None

    history_val_file = model_dir / "history_val.json"
    history_train_file = model_dir / "history_train.json"

    if not history_val_file.exists():
        return None

    try:
        with open(history_val_file, 'r') as f:
            val_history = json.load(f)

        with open(history_train_file, 'r') as f:
            train_history = json.load(f)

        # 检测任务类型
        if 'mae' in val_history:
            task_type = 'regression'
            metric_name = 'mae'
            val_metrics = val_history['mae']
            best_val = min(val_metrics)
            best_epoch = val_metrics.index(best_val)
        elif 'accuracy' in val_history:
            task_type = 'classification'
            metric_name = 'accuracy'
            val_metrics = val_history['accuracy']
            best_val = max(val_metrics)
            best_epoch = val_metrics.index(best_val)
        else:
            return None

        # 提取关键指标
        result = {
            'task_type': task_type,
            'metric_name': metric_name,
            'total_epochs': len(val_history['epochs']),
            'best_epoch': val_history['epochs'][best_epoch],
            'best_val': best_val,
            'final_val': val_metrics[-1],
            'best_train_loss': train_history['loss'][best_epoch],
            'final_train_loss': train_history['loss'][-1],
        }

        # 添加额外指标（如果存在）
        if task_type == 'regression':
            if 'rmse' in val_history:
                result['best_val_rmse'] = min(val_history['rmse'])
                result['final_val_rmse'] = val_history['rmse'][-1]
        elif task_type == 'classification':
            if 'precision' in val_history:
                result['best_val_precision'] = max(val_history['precision'])
                result['final_val_precision'] = val_history['precision'][-1]
            if 'recall' in val_history:
                result['best_val_recall'] = max(val_history['recall'])
                result['final_val_recall'] = val_history['recall'][-1]
            if 'f1' in val_history:
                result['best_val_f1'] = max(val_history['f1'])
                result['final_val_f1'] = val_history['f1'][-1]

        return result

    except Exception as e:
        print(f"警告: 读取 {model_dir} 时出错: {e}")
        return None


def summarize_results(base_dir, output_file=None, full_model_dir=None):
    """汇总所有实验结果（包括Full Model）"""

    base_dir = Path(base_dir)

    # 自动检测Full Model目录
    if full_model_dir is None:
        full_model_dir = base_dir.parent / "full_model_multi_seed"
    else:
        full_model_dir = Path(full_model_dir)

    # 实验配置
    exp_configs = {
        1: {
            'name': 'Exp-1: Baseline',
            'short_name': 'Baseline',
            'description': 'Text Simple Concat (no cross-modal attention)',
            'use_cross_modal': False,
            'use_middle_fusion': False,
            'use_fine_grained': False,
        },
        2: {
            'name': 'Exp-2: +Late Fusion',
            'short_name': '+Late',
            'description': 'Late fusion with cross-modal attention',
            'use_cross_modal': True,
            'use_middle_fusion': False,
            'use_fine_grained': False,
        },
        3: {
            'name': 'Exp-3: +Middle Fusion',
            'short_name': '+Middle',
            'description': 'Late + Middle fusion (Innovation 1)',
            'use_cross_modal': True,
            'use_middle_fusion': True,
            'use_fine_grained': False,
        },
        4: {
            'name': 'Exp-4: +Fine-Grained',
            'short_name': '+FineGrained',
            'description': 'Late + Fine-grained attention (Innovation 2)',
            'use_cross_modal': True,
            'use_middle_fusion': False,
            'use_fine_grained': True,
        },
        5: {
            'name': 'Exp-5: Full Model',
            'short_name': 'Full',
            'description': 'All innovations combined',
            'use_cross_modal': True,
            'use_middle_fusion': True,
            'use_fine_grained': True,
        },
    }

    seeds = [42, 123, 7]

    # 收集所有结果
    all_results = []
    task_type = None
    metric_name = None

    print("="*80)
    print("📊 多种子消融实验结果汇总（包括Full Model）")
    print("="*80)
    print(f"\n消融实验目录: {base_dir}")
    print(f"Full Model目录: {full_model_dir}\n")

    for exp_num in range(1, 6):
        config = exp_configs[exp_num]

        print(f"\n{config['name']}")
        print("-" * 60)

        exp_results = []

        for seed in seeds:
            # Exp-5 (Full Model) 可以从消融目录或独立目录加载
            if exp_num == 5:
                # 首先尝试从消融实验目录加载（exp5_seed格式）
                result = load_experiment_results(base_dir, exp_num, seed)

                # 如果没有找到，尝试从Full Model独立目录加载（full_model_seed格式）
                if result is None:
                    model_dir = full_model_dir / f"full_model_seed{seed}"
                    if model_dir.exists():
                        result = load_full_model_results(model_dir)
            else:
                # Exp-1到Exp-4从消融实验目录加载
                result = load_experiment_results(base_dir, exp_num, seed)

            if result is not None:
                if task_type is None:
                    task_type = result['task_type']
                    metric_name = result['metric_name']

                exp_results.append(result)

                # 打印单个种子的结果
                print(f"  Seed {seed:3d}: "
                      f"{metric_name}={result['best_val']:.4f} "
                      f"(epoch {result['best_epoch']}, "
                      f"total {result['total_epochs']} epochs)")
            else:
                print(f"  Seed {seed:3d}: 未完成或数据缺失")

        if exp_results:
            # 计算统计量
            best_vals = [r['best_val'] for r in exp_results]
            mean_val = np.mean(best_vals)
            std_val = np.std(best_vals, ddof=1) if len(best_vals) > 1 else 0

            print(f"\n  统计: {metric_name} = {mean_val:.4f} ± {std_val:.4f}")
            print(f"  完成数: {len(exp_results)}/{len(seeds)}")

            # 添加到总结果
            summary = {
                'exp_num': exp_num,
                'exp_name': config['name'],
                'short_name': config['short_name'],
                'description': config['description'],
                'use_cross_modal': config['use_cross_modal'],
                'use_middle_fusion': config['use_middle_fusion'],
                'use_fine_grained': config['use_fine_grained'],
                'num_completed': len(exp_results),
                'mean_best_val': mean_val,
                'std_best_val': std_val,
                'individual_results': exp_results,
            }

            all_results.append(summary)

    # ========================================================================
    # 生成CSV报告
    # ========================================================================
    print("\n" + "="*80)
    print("📄 生成CSV报告")
    print("="*80)

    if not all_results:
        print("\n❌ 没有可用的结果数据！")
        return

    # CSV 1: 简明汇总（均值±标准差）
    csv_rows = []
    for summary in all_results:
        row = {
            'Experiment': summary['short_name'],
            'Description': summary['description'],
            'Cross-Modal': '✓' if summary['use_cross_modal'] else '✗',
            'Middle Fusion': '✓' if summary['use_middle_fusion'] else '✗',
            'Fine-Grained': '✓' if summary['use_fine_grained'] else '✗',
            'Completed': f"{summary['num_completed']}/3",
            f'Best {metric_name.upper()} (Mean±Std)':
                f"{summary['mean_best_val']:.4f}±{summary['std_best_val']:.4f}",
        }
        csv_rows.append(row)

    df_summary = pd.DataFrame(csv_rows)

    summary_csv = base_dir / "ablation_summary.csv"
    df_summary.to_csv(summary_csv, index=False)
    print(f"\n✅ 简明汇总已保存: {summary_csv}")

    # CSV 2: 详细结果（包含每个种子）
    detailed_rows = []
    for summary in all_results:
        for i, seed in enumerate(seeds):
            if i < len(summary['individual_results']):
                result = summary['individual_results'][i]
                row = {
                    'Experiment': summary['short_name'],
                    'Seed': seed,
                    'Total Epochs': result['total_epochs'],
                    'Best Epoch': result['best_epoch'],
                    f'Best Val {metric_name.upper()}': result['best_val'],
                    f'Final Val {metric_name.upper()}': result['final_val'],
                    'Best Train Loss': result['best_train_loss'],
                    'Final Train Loss': result['final_train_loss'],
                }

                # 添加额外指标
                if task_type == 'regression' and 'best_val_rmse' in result:
                    row['Best Val RMSE'] = result['best_val_rmse']
                    row['Final Val RMSE'] = result['final_val_rmse']
                elif task_type == 'classification':
                    if 'best_val_precision' in result:
                        row['Best Val Precision'] = result['best_val_precision']
                    if 'best_val_recall' in result:
                        row['Best Val Recall'] = result['best_val_recall']
                    if 'best_val_f1' in result:
                        row['Best Val F1'] = result['best_val_f1']

                detailed_rows.append(row)

    df_detailed = pd.DataFrame(detailed_rows)

    detailed_csv = base_dir / "ablation_detailed.csv"
    df_detailed.to_csv(detailed_csv, index=False)
    print(f"✅ 详细结果已保存: {detailed_csv}")

    # ========================================================================
    # 打印对比表格
    # ========================================================================
    print("\n" + "="*80)
    print("📊 性能对比表")
    print("="*80)
    print()

    print(df_summary.to_string(index=False))

    # ========================================================================
    # 分析改进效果
    # ========================================================================
    print("\n" + "="*80)
    print("📈 改进效果分析")
    print("="*80)
    print()

    if len(all_results) >= 2:
        baseline = all_results[0]

        print(f"基线 (Baseline): {baseline['mean_best_val']:.4f} ± {baseline['std_best_val']:.4f}")
        print()

        for summary in all_results[1:]:
            improvement = baseline['mean_best_val'] - summary['mean_best_val']
            improvement_pct = (improvement / baseline['mean_best_val']) * 100

            if task_type == 'regression':
                direction = "降低" if improvement > 0 else "增加"
            else:
                direction = "提升" if improvement > 0 else "下降"

            print(f"{summary['short_name']:20s}: "
                  f"{summary['mean_best_val']:.4f} ± {summary['std_best_val']:.4f} "
                  f"→ {direction} {abs(improvement):.4f} ({abs(improvement_pct):.2f}%)")

        print()

        # 最佳模型
        if task_type == 'regression':
            best_exp = min(all_results, key=lambda x: x['mean_best_val'])
        else:
            best_exp = max(all_results, key=lambda x: x['mean_best_val'])

        print(f"🏆 最佳配置: {best_exp['short_name']} "
              f"({metric_name.upper()} = {best_exp['mean_best_val']:.4f} ± {best_exp['std_best_val']:.4f})")

    print("\n" + "="*80)
    print("✅ 汇总完成！")
    print("="*80)
    print()


def main():
    parser = argparse.ArgumentParser(description='多种子消融实验结果汇总（包括Full Model）')
    parser.add_argument('--ablation_dir', type=str, default='./ablation_multi_seed',
                        help='消融实验基础目录')
    parser.add_argument('--full_model_dir', type=str, default=None,
                        help='Full Model训练目录（可选，默认自动检测为 ../full_model_multi_seed）')
    parser.add_argument('--output', type=str, default=None,
                        help='输出CSV文件路径（可选）')

    args = parser.parse_args()

    if not os.path.exists(args.ablation_dir):
        print(f"错误: 目录不存在: {args.ablation_dir}")
        sys.exit(1)

    summarize_results(args.ablation_dir, args.output, args.full_model_dir)


if __name__ == '__main__':
    main()
