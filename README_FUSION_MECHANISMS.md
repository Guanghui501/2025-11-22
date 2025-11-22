# 多模态融合机制详解 - 快速参考

## 📚 文档索引

1. **详细文档**: `MODAL_FUSION_MECHANISMS.md` - 完整的技术文档
2. **可视化脚本**: `visualize_fusion_flow.py` - 生成流程图和特征变化图
3. **生成的图片**:
   - `fusion_mechanisms_timeline.pdf/png` - 融合机制时间线
   - `feature_dimensions_comparison.pdf/png` - 特征维度变化对比
   - `attention_heatmap_example.pdf/png` - 细粒度注意力示例

---

## 🎯 核心问题回答

### Q: "only text" 是哪个阶段的文本特征？

**A**: 是经过以下处理的文本特征：
```
文本输入 → MatSciBERT编码 → 提取CLS token → 投影到64维 → (可选)跨模态注意力增强
[batch, 768] → [batch, 64] → [batch, 64]
```

- **维度**: `[batch_size, 64]`
- **位置**: 在最终融合**之前**的独立文本表示
- **代码位置**: `alignn.py:878-880`, `visualize_latent_space.py:128-129`

---

## 🔄 三大融合机制总览

| 机制 | 作用时机 | 粒度 | 方向 | 维度变化 | 核心作用 |
|-----|---------|-----|-----|---------|---------|
| **中间融合** | ALIGNN第2层 | 原子级 | 文→图 | ❌ 不变 | 文本调制图编码 |
| **细粒度注意力** | 图编码后 | 原子-词级 | 双向 | ❌ 不变 | 精细语义匹配 |
| **全局注意力** | 最终融合前 | 全局 | 双向 | ❌ 不变 | 深度特征融合 |

---

## 📊 特征变化流程

### 1. 中间融合 (Middle Fusion)

```
第2层ALIGNN后: 节点特征 [total_atoms, 256]
                    ↓
            文本特征 [batch, 64] → 变换到256维 → 广播到每个原子
                    ↓
            门控机制: gate = σ(concat([节点, 文本]))
                    ↓
增强的节点特征: node + gate * text_broadcasted
```

**关键点**:
- 每个原子自适应决定接受多少文本信息
- 文本→图的单向调制

**示例**:
```
文本: "copper oxide superconductor"
Cu原子: gate=0.9 → 强烈接受"copper"信息
O原子:  gate=0.7 → 接受"oxide"信息
Fe原子: gate=0.1 → 弱接受（文本未提及）
```

---

### 2. 细粒度注意力 (Fine-grained Attention)

```
原子特征: [B, max_atoms, 256] ──┐
                              原子→词注意力
文本token: [B, seq_len, 768] ──┘
                ↓
         增强的原子特征 [B, max_atoms, 256]
         增强的token特征 [B, seq_len, 768]
```

**双向注意力**:
- **原子→词**: 每个原子关注文本中的哪些词？
- **词→原子**: 每个词关注结构中的哪些原子？

**注意力矩阵示例**:
```
        Silicon  dioxide  thermal
Si原子    0.7     0.2      0.1    ← Si关注"Silicon"
O原子     0.1     0.6      0.2    ← O关注"dioxide"
```

**可解释性**: 可以可视化原子-词的对应关系

---

### 3. 全局注意力 (Cross-modal Attention)

```
图特征: [B, 64] ──┐
                全局双向注意力
文本特征: [B, 64] ──┘
        ↓
    增强的图特征 [B, 64]
    增强的文本特征 [B, 64]
        ↓
    平均融合: (图' + 文本') / 2
        ↓
    最终预测
```

**关键点**:
- 全局级别的深度交互
- 两个模态完全对称地相互增强

---

## 🎨 可视化中的特征对应

在 `visualize_latent_space.py` 中提取的三种特征：

### `'graph'` 特征
```python
graph_features = output['graph_features']  # [batch, 64]
```
**来源**:
- 基础: 图池化 + 投影
- 如果启用中间融合: 包含文本调制的影响
- 如果启用细粒度注意力: 包含原子-词交互的影响
- 如果启用全局注意力: 返回 `enhanced_graph`

### `'text'` 特征
```python
text_features = output['text_features']  # [batch, 64]
```
**来源**:
- 基础: CLS token + 投影
- 如果启用细粒度注意力: 包含词-原子交互的影响
- 如果启用全局注意力: 返回 `enhanced_text`

### `'fused'` 特征
```python
fused = np.concatenate([graph_features, text_features], axis=1)  # [batch, 128]
```
**来源**: 简单拼接上述两者

---

## 🎯 各机制对比总结

| 机制 | 融合时机 | 融合粒度 | 方向 | 特征维度变化 | 主要作用 |
|-----|---------|---------|-----|-------------|---------|
| **中间融合** | ALIGNN层内 | 原子级 | 单向(文→图) | ❌ 不变 | 文本调制图编码 |
| **细粒度注意力** | 图编码后 | 原子-词级 | 双向 | ❌ 不变 | 精细语义对齐 |
| **全局注意力** | 最终融合 | 全局 | 双向 | ❌ 不变 | 全局信息融合 |

---

## 🔬 特征空间演化

### 无交互（基础模型）
```
图空间:     ●  ●  ●  ●  ●
                ↓
文本空间:   ○  ○  ○  ○  ○
           (两个独立空间)
```

### 中间融合后
```
文本→图:    ●' ●' ●' ●' ●'
            ↑文本增强的图特征
文本:       ○  ○  ○  ○  ○
           (文本未变)
```

### 细粒度注意力后
```
互相增强:   ●" ●" ●" ●" ●"
            ↕️  ↕️  ↕️  ↕️
            ○" ○" ○" ○"
```

### 全局注意力后
```
最终融合:   ⊕  ⊕  ⊕  ⊕  ⊕
          (图文深度融合的统一表示)
```

---

## 💡 实际应用建议

### 选择融合策略

1. **需要文本引导图编码**:
   - 启用: `use_middle_fusion=True`
   - 适合: 文本描述对结构理解很重要的任务

2. **需要可解释性**:
   - 启用: `use_fine_grained_attention=True`
   - 适合: 需要知道哪些原子对应哪些词

3. **最大融合效果**:
   - 同时启用所有三个机制
   - 适合: 复杂的多模态推理任务

---

## 📝 代码位置索引

| 功能 | 文件 | 行号 |
|-----|------|-----|
| 中间融合定义 | `alignn.py` | 121-218 |
| 中间融合应用 | `alignn.py` | 896-899 |
| 细粒度注意力定义 | `alignn.py` | 352-528 |
| 细粒度注意力应用 | `alignn.py` | 905-952 |
| 全局注意力定义 | `alignn.py` | 221-349 |
| 全局注意力应用 | `alignn.py` | 960-971 |
| 特征提取 | `visualize_latent_space.py` | 57-157 |

---

## 🎓 关键概念

### 1. 融合时机
- **早期融合** (Middle): 在编码过程中融合
- **晚期融合** (Cross-modal): 编码完成后融合
- **混合融合**: 结合早期和晚期

### 2. 融合粒度
- **全局**: 整个图/整个文本 (全局注意力)
- **局部**: 原子级/词级 (细粒度注意力)
- **中间**: 节点级 (中间融合)

### 3. 融合方向
- **单向**: 文本→图 (中间融合)
- **双向**: 图↔文本 (细粒度注意力、全局注意力)

---

## ⚡ 快速验证

查看模型使用了哪些融合机制:
```python
checkpoint = torch.load('best_val_model.pt')
config = checkpoint['config']

print(f"中间融合: {config.use_middle_fusion}")
print(f"细粒度注意力: {config.use_fine_grained_attention}")
print(f"全局注意力: {config.use_cross_modal_attention}")
```

提取注意力权重进行可视化:
```python
output = model(input, return_attention=True)
global_attn = output['attention_weights']
fine_attn = output['fine_grained_attention_weights']

# 可视化原子-词注意力
atom_to_text = fine_attn['atom_to_text']  # [batch, heads, atoms, tokens]
```

---

**文档更新时间**: 2025-11-22
