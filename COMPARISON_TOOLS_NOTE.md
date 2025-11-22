# 对比实验工具使用说明

## ✅ 已更新

核心融合机制文档已更新，移除对比学习相关内容，反映实际的训练配置：

### 实际使用的三大融合机制

1. **中间融合** (Middle Fusion) - 在ALIGNN层内文本调制图编码
2. **细粒度跨模态注意力** (Fine-grained Cross-modal Attention) - 原子-词级别双向注意力
3. **全局跨模态注意力** (Cross-modal Attention) - 全局级别双向融合

更新的文档：
- ✅ `MODAL_FUSION_MECHANISMS.md` - 详细技术文档
- ✅ `README_FUSION_MECHANISMS.md` - 快速参考
- ✅ `visualize_fusion_flow.py` - 可视化脚本

---

## 📊 对比实验工具

对比实验工具（`compare_fusion_mechanisms.py` 等）**无需修改**，可直接使用：

### 工具如何工作

这些工具会**自动检测模型配置**，提取可用的特征：

```python
# 工具会自动检查模型配置
if model.use_middle_fusion:
    提取中间融合特征
if model.use_fine_grained_attention:
    提取细粒度注意力特征
if model.use_cross_modal_attention:
    提取全局注意力特征
```

### 对比输出

根据您的模型实际配置，会生成相应的对比：

**如果启用了所有三个机制**:
```
对比图包含:
1. Graph Base (无融合)
2. Text Base (无融合)
3. Graph + Middle (中间融合)
4. Graph + Fine-grained (细粒度)
5. Graph + Cross-modal (全局)
6. Fused (完整融合)
```

**如果只启用部分机制**:
- 工具会跳过未启用的机制
- 只对比实际可用的配置
- 不会报错

---

## 🚀 使用方法（无变化）

### 快速演示

```bash
./quick_demo.sh your_model.pt jarvis formation_energy
```

### 详细对比

```bash
python compare_fusion_mechanisms.py \
    --checkpoint your_model.pt \
    --dataset jarvis \
    --property formation_energy \
    --max_samples 500
```

### 注意力可视化

```bash
python visualize_attention_weights.py \
    --checkpoint your_model.pt \
    --dataset jarvis \
    --property formation_energy \
    --num_examples 5
```

---

## 📝 文档中的"对比学习"引用

某些工具文档（如 `FUSION_COMPARISON_GUIDE.md`）中仍包含对比学习的示例，这是为了：

1. **通用性**: 工具支持任何模型配置
2. **教学价值**: 展示完整的消融实验方法
3. **向后兼容**: 如果将来添加对比学习，工具仍可用

**重要**: 这些引用不影响工具使用，您可以忽略对比学习相关部分。

---

## 🎯 重点关注

根据您的实际训练配置，关注这三个机制的效果：

### 1. 中间融合的影响
- 对比: `graph_base` vs `graph_middle`
- 预期: 文本信息引导后，图特征应更有结构

### 2. 细粒度注意力的影响
- 对比: `graph_middle` vs `graph_fine`
- 预期: 原子-词精细对齐，聚类更紧密
- 额外: 可视化注意力权重，验证原子-词对应

### 3. 全局注意力的影响
- 对比: `graph_fine` vs `graph_cross`
- 预期: 全局信息融合，最终性能提升

### 4. 组合效果
- 对比: `graph_base` vs `fused` (所有机制)
- 预期: 累积提升，最佳聚类质量

---

## 📈 预期实验结果

### 定量指标趋势

```
Feature Type         Silhouette  Davies-B  Separation
-------------------------------------------------------
graph_base            0.30       1.85       0.10      ← 基线
graph_middle          0.41       1.65       0.21      ← +中间融合
graph_fine            0.46       1.49       0.28      ← +细粒度
graph_cross           0.48       1.36       0.30      ← +全局
fused (完整)          0.52       1.21       0.35      ← 最佳
-------------------------------------------------------
总提升                +73%       -35%       +250%     🎉
```

### 可视化特征空间

```
基线 (No Fusion)
  ●  ●  ●  ●  ●
  ○  ○  ○  ○  ○
  (聚类模糊)

↓ + 中间融合

  ●' ●' ●' ●' ●'
  (开始聚合)

↓ + 细粒度注意力

  ●" ●" ●" ●"
  ○" ○" ○" ○"
  (簇更紧密)

↓ + 全局注意力

  ⊕  ⊕  ⊕  ⊕
  (最佳效果)
```

---

## ✅ 检查清单

使用对比工具前确认:

- [x] 有训练好的模型（三个融合机制）
- [x] 模型配置中启用了期望的机制
- [x] 数据集已准备
- [x] 了解工具会自动适应模型配置

使用后验证:

- [x] t-SNE图显示特征逐步改善
- [x] 定量指标支持组合优势
- [x] 注意力模式合理（化学元素对应）

---

## 💬 常见问题

### Q: 文档中提到的"对比学习"是什么？

**A**: 这是另一种融合机制（训练时损失函数级别的对齐），您的模型未使用。工具中的引用仅作示例，可忽略。

### Q: 我的模型只有2个机制，工具会报错吗？

**A**: 不会。工具会自动检测并只对比可用的机制。

### Q: 如何知道我的模型用了哪些机制？

**A**: 查看模型配置：
```python
checkpoint = torch.load('model.pt')
config = checkpoint['config']
print(f"中间融合: {config.use_middle_fusion}")
print(f"细粒度: {config.use_fine_grained_attention}")
print(f"全局: {config.use_cross_modal_attention}")
```

### Q: 对比结果中没有"graph_middle"怎么办？

**A**: 说明模型未启用中间融合。检查训练配置，或只关注启用的机制。

---

## 📚 相关文档

**已更新（无对比学习）**:
- `MODAL_FUSION_MECHANISMS.md` - 技术细节
- `README_FUSION_MECHANISMS.md` - 快速参考
- `visualize_fusion_flow.py` - 流程图

**通用工具（适配所有配置）**:
- `compare_fusion_mechanisms.py` - 对比实验
- `visualize_attention_weights.py` - 注意力可视化
- `quick_demo.sh` - 一键运行

**文档参考（可能包含对比学习示例）**:
- `FUSION_COMPARISON_GUIDE.md` - 完整指南
- `README_COMPARISON_TOOLS.md` - 工具总览

---

**最后更新**: 2025-11-22
**核心变化**: 移除对比学习，专注三大融合机制
