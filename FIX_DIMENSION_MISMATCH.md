# 修复: 特征提取时的维度不匹配错误

## 问题描述

运行 `compare_fusion_mechanisms.py` 时出现错误:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x128 and 64x64)
```

错误发生在 `models/alignn.py:976` 的 `h = F.relu(self.fc1(h))` 行。

## 根本原因

模型的 `fc1` 层在初始化时根据 `use_cross_modal_attention` 的值设置维度:
- 如果 `use_cross_modal_attention = True`: `fc1 = Linear(64, 64)` (因为使用平均融合)
- 如果 `use_cross_modal_attention = False`: `fc1 = Linear(128, 64)` (因为使用拼接)

旧的 `compare_fusion_mechanisms.py` 尝试通过动态修改 `model.use_cross_modal_attention` 来进行消融实验,但这会导致:
- 模型架构 (`fc1` 维度) 是固定的
- 运行时改变 `use_cross_modal_attention` 会让模型进入错误的分支
- 导致维度不匹配

**简单来说**: 我们不能在运行时改变影响模型架构的配置参数!

## 解决方案

### 1. 修改模型 (`models/alignn.py`)

添加 `return_intermediate_features` 参数到 `forward()` 函数:

```python
def forward(self, g, return_features=False, return_attention=False,
            return_intermediate_features=False):
```

在前向传播的关键位置收集中间特征:
- `text_emb_base`: 文本投影后的基础特征 (融合前)
- `graph_emb_base`: 图投影后的基础特征 (融合前)
- `enhanced_graph`, `enhanced_text`: 全局注意力后的特征 (如果启用)

返回这些中间特征用于消融研究,**而不改变模型架构或配置**。

### 2. 重写比较脚本 (`compare_fusion_mechanisms.py`)

新版本使用 `return_intermediate_features=True` 参数:

```python
# 旧方法 (错误):
self.model.use_cross_modal_attention = False  # ❌ 改变配置会破坏架构!
output = self.model(input, return_features=True)

# 新方法 (正确):
output = self.model(input, return_intermediate_features=True)  # ✅
# 输出包含: graph_base, text_base, graph_cross, text_cross, 等等
```

新脚本提取的特征:
- `graph_base`, `text_base`: 融合前的基础特征
- `graph_cross`, `text_cross`: 全局注意力后的特征 (如果启用)
- `graph_final`, `text_final`: 最终特征
- `fused`: 最终融合特征

### 3. 保留旧脚本作为备份

```bash
compare_fusion_mechanisms_old.py  # 旧版本 (有bug)
compare_fusion_mechanisms.py      # 新版本 (已修复)
```

## 使用方法

现在可以正常运行:

```bash
python compare_fusion_mechanisms.py \
    --checkpoint outputs/your_model.pt \
    --dataset jarvis \
    --property formation_energy_peratom \
    --save_dir ./comparison_results
```

脚本会自动:
1. 检测模型启用了哪些融合机制
2. 提取对应阶段的特征
3. 生成 t-SNE 可视化
4. 计算特征质量指标

## 关键改进

1. **不再破坏模型架构**: 不动态修改 `use_*` 配置参数
2. **提取真实的中间特征**: 在前向传播中收集实际的中间状态
3. **自动适配模型配置**: 脚本自动检测启用了哪些机制
4. **更简洁的代码**: 只需一次前向传播即可获取所有特征

## 技术要点

**为什么不能动态修改融合配置?**

模型初始化时会根据配置创建固定的层:
```python
if use_cross_modal_attention:
    self.fc1 = nn.Linear(64, 64)  # 期望 64维 输入
else:
    self.fc1 = nn.Linear(128, 64)  # 期望 128维 输入
```

如果训练时 `use_cross_modal_attention=True`, 则 `fc1` 期望 64维。
运行时设置为 `False` 会执行拼接分支产生 128维, 导致维度不匹配。

**正确的消融研究方法:**

不是关闭融合机制, 而是**提取融合前后的特征**来对比效果。这样既不破坏架构, 又能看到融合的影响。

## 注意事项

- `visualize_attention_weights.py` 不受此问题影响 (它只使用 `return_attention=True`)
- 所有多模型比较脚本 (`MULTI_MODEL_COMPARISON_GUIDE.md`) 仍然适用
- 只需要用新的 `compare_fusion_mechanisms.py` 即可

## 相关文件

- `models/alignn.py`: 添加了 `return_intermediate_features` 参数
- `compare_fusion_mechanisms.py`: 完全重写,使用新方法
- `compare_fusion_mechanisms_old.py`: 旧版本备份
