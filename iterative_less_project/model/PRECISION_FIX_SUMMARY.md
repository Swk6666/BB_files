# LESS算法精度修复总结

## 问题原因
您的代码与原论文LESS代码在精度处理上存在关键差异，导致梯度计算出现NaN/Inf。

## 原论文的精度体系
根据对原论文代码的详细分析：

### 1. 模型加载精度
- **位置**: `LESS/less/data_selection/get_info.py` 第23、66、99行
- **设置**: `torch_dtype=torch.bfloat16` (默认)
- **结果**: 模型参数为bfloat16

### 2. 优化器状态精度
- **位置**: `LESS/less/data_selection/get_info.py` 第128-130行
- **加载**: `torch.load(optimizer_path)["state"]`
- **精度**: 保持优化器保存时的原始精度（通常为float32）

### 3. Adam梯度计算精度
- **位置**: `LESS/less/data_selection/collect_grad_reps.py` 第111-127行
- **方式**: 直接混合精度计算（bfloat16梯度 + float32优化器状态）
- **无额外精度转换**

### 4. 影响力分数计算精度 ⭐ **关键发现**
- **位置**: `LESS/less/data_selection/matching.py` 第64、73行
- **强制转换**: `validation_info.to(device).float()` 和 `training_info.to(device).float()`
- **结果**: 所有计算在float32精度下进行

## 修复内容

### 1. 梯度计算逻辑修复
**文件**: `externals/less/data_selection/collect_grad_reps.py`

```python
# 完全按照原论文obtain_gradients_with_adam函数
if gradient_type == "adam":
    beta1 = 0.9
    beta2 = 0.999  
    eps = 1e-08
    
    vectorized_grads = torch.cat([p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None])
    
    updated_avg = beta1 * m_global + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * v_global + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)
```

### 2. 影响力分数计算修复 ⭐ **最关键**
**文件**: `iterative_less/toolkit.py`

```python
# 【关键修复】强制转换为float32，完全按照原论文matching.py
train_grads = train_grads.float()  # 原论文第73行
val_grads = val_grads.float()      # 原论文第64行

# 【原论文标准L2归一化】
train_grads_norm = F.normalize(train_grads, p=2, dim=1)
val_grads_norm = F.normalize(val_grads, p=2, dim=1)

# 【原论文影响力计算】
similarity_matrix = torch.matmul(train_grads_norm, val_grads_norm.T)
final_scores = similarity_matrix.max(dim=1)[0]
```

### 3. 优化器加载逻辑修复
**文件**: `externals/less/data_selection/get_info.py`

```python
# 按照原论文，优先尝试optimizer.bin，然后尝试optimizer.pt
# 完全按照原论文方式加载: torch.load(optimizer_path)["state"]
```

## 精度一致性验证

### 修复前
- 梯度计算：混合精度但处理不当
- 影响力计算：可能仍在bfloat16下进行
- 结果：NaN/Inf导致分数为0

### 修复后
- 梯度计算：完全按照原论文的混合精度方式
- 影响力计算：**强制float32**（这是关键！）
- 结果：应该产生有效的非零影响力分数

## 期望结果
修复后，您应该看到：
1. ✅ 影响力分数不再为NaN或全零
2. ✅ 分数统计信息显示合理的数值范围
3. ✅ 算法能够正常选择样本进行训练

## 下一步建议
1. 重新运行 `main_less.py`
2. 观察日志中的分数统计信息
3. 如果仍有问题，可以运行 `debug_gradients.py` 进行详细分析
