# 训练代码逻辑验证报告

## 用户说的逻辑

1. **如果分段点距离要分段点超过4句话，就使用固定窗口（取最后4句）**
2. **自适应窗口只会减少前文，不会缩短上下文，最多缩短前文**

## 代码逻辑分析

### 关键代码段（train.py 第98-124行）

```python
if self.adaptive_context:
    nearest_segment = self._find_nearest_segment_before(original_idx, segment_positions)
    
    if nearest_segment is not None:
        context_start_idx = nearest_segment + 1
    else:
        context_start_idx = max(0, original_idx - self.context_sentences)
    
    window_context_start = max(0, context_start_idx - start_idx)
    window_context_end = i
    
    context_before = window_sentences[window_context_start:window_context_end]
    if len(context_before) > self.context_sentences:
        context_before = context_before[-self.context_sentences:]
```

### 逻辑验证

#### 逻辑1验证：分段点距离超过4句时使用固定窗口

**测试场景：**
- `context_sentences = 4`
- `original_idx = 20`（当前位置）
- `i = 10`（窗口内位置）
- `start_idx = 10`（窗口起始位置）

**测试结果：**

| 分段点位置 | 距离 | 前文长度 | 是否截断 | 是否符合逻辑1 |
|-----------|------|---------|---------|--------------|
| 19 | 1句 | 0句 | 否 | ✓ 从分段点后开始（自适应优势）|
| 18 | 2句 | 1句 | 否 | ✓ 从分段点后开始（自适应优势）|
| 17 | 3句 | 2句 | 否 | ✓ 从分段点后开始（自适应优势）|
| 16 | 4句 | 3句 | 否 | ✓ 从分段点后开始（自适应优势）|
| 15 | 5句 | 4句 | 否 | ✓ 效果等同于固定窗口 |
| 12 | 8句 | 4句 | 是 | ✓ **取最后4句（等同于固定窗口）** |
| 5 | 15句 | 4句 | 是 | ✓ **取最后4句（等同于固定窗口）** |

**结论：** ✓ **代码逻辑符合用户说的逻辑1**

- 当分段点距离 ≤ 4句时，从分段点后开始取上下文（可能少于4句，这是自适应窗口的优势）
- 当分段点距离 > 4句时，会截断到最后4句，效果等同于固定窗口

#### 逻辑2验证：后文不受自适应窗口影响

**代码第124行：**
```python
context_after = window_sentences[i + 2:min(len(window_sentences), i + 2 + self.context_sentences)]
```

**分析：**
- `context_after` 的计算完全独立于 `adaptive_context` 条件
- 无论 `adaptive_context` 是 `True` 还是 `False`，后文都是固定的
- 后文长度 = `min(window_length - (i+2), context_sentences)`

**结论：** ✓ **代码逻辑符合用户说的逻辑2**

- 自适应窗口只影响前文（`context_before`）
- 后文（`context_after`）始终保持固定长度，不受自适应窗口影响

## 总结

### ✓ 代码逻辑完全符合用户说的逻辑

1. **逻辑1正确实现：**
   - 分段点距离 > `context_sentences` 时，自动截断到最后 `context_sentences` 句（等同于固定窗口）
   - 分段点距离 ≤ `context_sentences` 时，从分段点后开始取上下文（自适应窗口优势）

2. **逻辑2正确实现：**
   - 后文完全独立于 `adaptive_context` 条件
   - 后文始终保持固定长度，不受自适应窗口影响

3. **自适应窗口的优势：**
   - 当分段点较近时（≤ `context_sentences` 句），避免引入前一段的无关上下文
   - 提升语义连贯性，减少跨段干扰

### 代码实现细节

- **第112行：** `window_context_start = max(0, context_start_idx - start_idx)` 确保索引不越界
- **第118-119行：** 当上下文超过 `context_sentences` 时，截断到最后 `context_sentences` 句
- **第124行：** 后文计算独立于自适应窗口逻辑

## 建议

虽然代码逻辑正确，但建议：

1. **训练/推理一致性：** `test.py` 中的 `predict_sentence_pairs` 函数也应实现相同的自适应窗口逻辑，确保训练和推理时上下文处理一致。

2. **添加注释：** 可以在代码中添加注释说明自适应窗口的工作机制，特别是第118-119行的截断逻辑。
