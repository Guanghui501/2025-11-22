# 下一步行动指南

## ✅ 已完成

### 1. 注意力权重对比工具（已提交并推送）

创建了完整的注意力权重分析工具，用于验证中期融合改善节点-文本对齐的假设。

**新增文件：**
- `compare_attention_weights.py` - 主分析脚本
- `run_attention_comparison.sh` - 快速执行脚本
- `README_ATTENTION_ANALYSIS.md` - 使用指南

**特性：**
- ✅ 灵活的checkpoint加载（支持多种格式）
- ✅ JARVIS数据集加载（与训练脚本一致）
- ✅ 4个关键指标计算（熵、最大权重、有效token数、基尼系数）
- ✅ 自动生成可视化（统计对比图 + 注意力热图）
- ✅ 详细的结果解读指南

### 2. 配置文件修复（已合并到config.py）

- ✅ ACGCNNConfig 导入改为可选
- ✅ ALIGNN_LN_Config 导入改为可选
- ✅ 训练脚本可以正常继续运行

### 3. Proposal C 实现（已完成）

- ✅ `use_text_fine_gate_fusion` - 使用text_fine门控调制graph
- ✅ `cross_attention_use_text_fine` - 全局注意力使用text_fine

## 🎯 当前状态

### 模型性能
```
全模态（中期+细粒度+全局）：MAE = 0.255
无中期（细粒度+全局）：    MAE = 0.274
性能提升：7.5%
```

### 待解答的问题
**为什么中期融合降低了特征的线性质量（-9.1% Pearson），但提升了最终预测性能？**

## 🚀 下一步操作

### 步骤 1：运行注意力权重分析

```bash
# 方法1：使用快速脚本（推荐）
bash run_attention_comparison.sh

# 方法2：手动指定参数
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

**预计时间**：取决于样本数和GPU，100个样本约5-10分钟

### 步骤 2：查看分析结果

```bash
# 查看统计数据
cat attention_analysis/attention_statistics.csv

# 查看可视化
ls attention_analysis/*.png
```

**预期看到：**
```
attention_analysis/
├── attention_statistics_comparison.png  # 4个指标的对比柱状图
├── attention_statistics.csv             # 详细数值数据
├── attention_heatmap_example_1.png      # 示例热图对比
├── attention_heatmap_example_2.png
└── attention_heatmap_example_3.png
```

### 步骤 3：解读结果

参考 `ATTENTION_ANALYSIS_GUIDE.md` 中的解读指南：

```bash
cat ATTENTION_ANALYSIS_GUIDE.md
```

#### 场景 1：假设得到验证 ✅

如果全模态模型的注意力指标显著优于无中期模型：
- 注意力熵 ↓ （降低20-30%）
- 最大权重 ↑ （提升30-50%）
- 有效Token数 ↓ （减少30-50%）
- 基尼系数 ↑ （提升20-40%）

**结论**：中期融合确实改善了节点特征质量，使细粒度注意力对齐更精准，这解释了性能提升的来源。

#### 场景 2：假设部分支持

指标有小幅改善（5-15%），但不明显。

**结论**：中期融合的作用是多方面的，不仅是改善注意力对齐，还包括特征多样性、泛化能力等。

#### 场景 3：意外发现

注意力指标相反，但性能仍然更好。

**结论**：模型学习了更复杂的对齐策略，不是简单集中注意力。

### 步骤 4：撰写论文

根据分析结果，在论文中添加消融研究部分：

**建议章节**：
```markdown
### 3.5 中期融合的机制分析

为了理解中期融合如何改善模型性能，我们对比了全模态模型和
无中期融合模型的细粒度注意力权重。

**注意力质量指标对比**

| 指标 | 全模态 | 无中期 | 改善 |
|------|--------|--------|------|
| 注意力熵 | X.XX | X.XX | -X% |
| 最大权重 | X.XX | X.XX | +X% |
| 有效Token数 | X.XX | X.XX | -X% |
| 基尼系数 | X.XX | X.XX | +X% |

如表所示，全模态模型的注意力分布更加集中...
[继续论述]

**图X：注意力热图对比**
[插入 attention_heatmap_example_1.png]

全模态模型的注意力热图呈现更明确的块状结构，而无中期融合
模型的注意力较为分散...
```

## 📊 其他可选分析

如果需要更深入的分析，可以扩展：

### 1. 统计显著性检验
```python
from scipy.stats import ttest_ind
# t检验验证差异的统计显著性
```

### 2. Case Study
选择几个典型样本，详细分析：
- 简单样本 vs 复杂样本
- 预测准确的样本 vs 预测错误的样本

### 3. 注意力-性能相关性
分析注意力熵与预测误差的相关性。

### 4. 按层分析
如果有多层细粒度注意力，对比各层的变化。

## 🔍 故障排除

### 如果遇到错误

1. **模型加载失败**
   - 检查checkpoint路径是否正确
   - 脚本会自动尝试多种格式，查看错误信息

2. **数据加载失败**
   - 确保 `cif/` 和 `description.csv` 存在
   - 检查 dataset 和 property 参数是否正确

3. **GPU内存不足**
   - 减小 `--batch_size`（默认64，可改为32或16）
   - 减小 `--num_samples`（默认100，可改为50）

4. **找不到函数或模块**
   - 确保在正确的Python环境中运行
   - 检查是否在项目根目录

## 📝 总结

```
当前任务：验证中期融合改善节点-文本对齐的假设

方法：对比全模态和无中期模型的细粒度注意力权重

预期：全模态模型的注意力更集中、更精准

意义：解释为什么中期融合虽然降低了线性特征质量，
     但仍然提升了最终预测性能（0.255 vs 0.274）
```

## 📚 相关文档

- `README_ATTENTION_ANALYSIS.md` - 使用指南（快速入门）
- `ATTENTION_ANALYSIS_GUIDE.md` - 详细分析指南（指标解释、结果解读）
- `compare_attention_weights.py` - 源代码（技术细节）

---

**现在可以开始运行分析了！**

```bash
bash run_attention_comparison.sh
```
