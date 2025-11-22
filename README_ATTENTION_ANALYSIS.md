# 注意力权重对比分析工具

## 📋 目的

对比**全模态模型**（中期+细粒度+全局）和**无中期融合模型**（细粒度+全局）的注意力权重，以验证中期融合如何改善节点-文本对齐。

## 🎯 核心假设

```
假设：中期融合改善了节点特征 → 细粒度注意力对齐更精准

全模态（有中期融合）:
├─ 节点已包含文本信息（通过中期融合）
├─ 细粒度注意力在更好的基础上对齐
└─ 预期：注意力更集中、更准确

无中期融合:
├─ 节点和文本从未交互
├─ 细粒度注意力需要从零学习对齐
└─ 预期：注意力更分散、更模糊
```

## 🚀 快速开始

### 1. 准备模型文件

确保你有两个训练好的模型：
- 全模态模型（包含中期融合）
- 无中期融合模型（只有细粒度+全局融合）

### 2. 运行分析

#### 方法 1：使用快速脚本（推荐）

```bash
bash run_attention_comparison.sh
```

**注意**：脚本中的路径是预配置的，如果你的路径不同，请修改脚本中的变量。

#### 方法 2：手动运行

```bash
python compare_attention_weights.py \
    --checkpoint_full output_100epochs_42_bs128_sw_ju/mbj_bandgap/best_val_model.pt \
    --checkpoint_no_middle output_100epochs_42_bs128_sw_ju_houqi/mbj_bandgap/best_val_model.pt \
    --root_dir /public/home/ghzhang/crysmmnet-main/dataset/jarvis/mbj_bandgap \
    --dataset dft_3d \
    --property mbj_bandgap \
    --save_dir ./attention_analysis \
    --num_samples 100 \
    --batch_size 64
```

### 3. 查看结果

分析会在 `attention_analysis/` 目录生成：

```
attention_analysis/
├── attention_statistics_comparison.png  # 统计指标对比图
├── attention_statistics.csv             # 详细统计数据
├── attention_heatmap_example_1.png      # 示例1的热图对比
├── attention_heatmap_example_2.png      # 示例2的热图对比
└── attention_heatmap_example_3.png      # 示例3的热图对比
```

## 📊 分析指标

| 指标 | 含义 | 期望（全模态） | 解释 |
|------|------|----------------|------|
| **注意力熵** | 分布的不确定性 | **更低** ↓ | 注意力更集中在关键token上 |
| **最大权重** | 最强对齐的权重值 | **更高** ↑ | 模型更确信重要的对齐 |
| **有效Token数** | 权重>0.1的token数 | **更少** ↓ | 更有选择性，不分散注意力 |
| **基尼系数** | 权重分布的不平等度 | **更高** ↑ | 少数token获得大部分注意力 |

## 🔍 结果解读

### 场景 1：假设验证（理想情况）

如果看到：
- 注意力熵：全模态 < 无中期（降低 20-30%）✅
- 最大权重：全模态 > 无中期（提升 30-50%）✅
- 有效Token数：全模态 < 无中期（减少 30-50%）✅
- 基尼系数：全模态 > 无中期（提升 20-40%）✅

**解释**：中期融合显著改善了细粒度注意力的对齐质量。节点在中期融合后已经"感知"到文本信息，因此细粒度注意力能更精准地找到关键的节点-token对应关系。

### 场景 2：假设部分支持

如果看到微小改善（5-15%），但最终性能仍然提升显著。

**解释**：中期融合的作用不仅仅是改善注意力对齐，还包括增加特征多样性、提升泛化能力等。

### 场景 3：意外发现

如果注意力指标与预期相反，但性能仍然更好。

**解释**：模型可能学习了更复杂的对齐策略，不是简单地集中注意力，而是学习多样化的对齐模式。

## 📝 论文写作建议

### 核心论述模板

```markdown
为了验证中期融合改善节点-文本对齐的机制，我们对比了全模态
模型和无中期融合模型的细粒度注意力权重。

结果显示，全模态模型的注意力分布显著更集中（熵降低X%），
最大权重更高（提升Y%），有效token数更少（减少Z%）。这表明
中期融合确实改善了节点特征的质量，使得细粒度注意力能够更
精准地识别节点-token的对应关系。

如图X所示，全模态模型的注意力热图呈现出更明确的块状结构，
而无中期融合模型的注意力较为分散。这一差异直接解释了为什么
全模态模型的预测性能更优（MAE 0.255 vs 0.274）。
```

## 🛠️ 命令行参数

```bash
python compare_attention_weights.py --help
```

主要参数：

- `--checkpoint_full`: 全模态模型的checkpoint路径
- `--checkpoint_no_middle`: 无中期融合模型的checkpoint路径
- `--root_dir`: 数据集根目录
- `--dataset`: 数据集类型（如 dft_3d, mp 等）
- `--property`: 属性名称（如 mbj_bandgap, formation_energy 等）
- `--save_dir`: 结果保存目录（默认：./attention_comparison）
- `--num_samples`: 分析的样本数（默认：100）
- `--batch_size`: 批大小（默认：32）

## 📚 相关文件

- `compare_attention_weights.py`: 主分析脚本
- `run_attention_comparison.sh`: 快速执行脚本
- `ATTENTION_ANALYSIS_GUIDE.md`: 详细分析指南

## ⚠️ 注意事项

1. **样本数量**：建议至少100个样本保证统计显著性
2. **批处理**：注意GPU内存，适当调整batch_size
3. **随机性**：使用固定的split_seed保证可复现性
4. **数据一致性**：确保两个模型在相同的测试集上对比

## 🔧 故障排除

### 问题 1：找不到模型文件

确保模型checkpoint路径正确，文件存在。

### 问题 2：数据加载失败

检查：
- 数据集根目录路径是否正确
- `cif/` 文件夹和 `description.csv` 是否存在
- dataset 和 property 参数是否与训练时一致

### 问题 3：GPU内存不足

减小 `--batch_size` 或 `--num_samples`。

### 问题 4：模型加载错误

脚本会自动尝试多种checkpoint格式（model_state_dict, state_dict, model等）。如果仍然失败，请检查checkpoint文件的内容。

## 🎯 预期结果示例

基于 mbj_bandgap 数据集的分析：

- 全模态模型测试集 MAE: 0.255
- 无中期融合模型测试集 MAE: 0.274
- 性能提升: 7.5%

注意力权重分析应该能解释这一性能差异的来源。

## 📖 详细文档

完整的分析指南、指标解释和结果解读，请参考：
```bash
cat ATTENTION_ANALYSIS_GUIDE.md
```
