# check_memory_usage.py
import os
import sys
import torch
from iterative_less import config
from externals.less.data_selection.get_training_dataset import get_training_dataset
from transformers import AutoTokenizer

# 模拟加载少量样本来估算单个样本大小
NUM_SAMPLES_TO_CHECK = 100
TOTAL_SAMPLES = 20000
MAX_SEQ_LENGTH = 1024 # 与您配置中一致

print("--- 开始内存占用估算 ---")
print(f"将加载 {NUM_SAMPLES_TO_CHECK} 个样本来估算单个样本的内存占用...")

tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, use_fast=True, local_files_only=True)
processed_dataset = get_training_dataset(
    config.CANDIDATE_POOL_FILENAME, 
    tokenizer, 
    MAX_SEQ_LENGTH,
    num_workers=1 # 强制单进程
)

# 加载到内存
subset_in_memory = [processed_dataset[i] for i in range(NUM_SAMPLES_TO_CHECK)]

# 计算单个样本的近似大小
single_sample_bytes = 0
if subset_in_memory:
    # 检查一个样本的所有tensor
    for key, tensor in subset_in_memory[0].items():
        if isinstance(tensor, torch.Tensor):
            single_sample_bytes += tensor.element_size() * tensor.nelement()

print(f"单个样本在内存中的近似大小: {single_sample_bytes / 1024:.2f} KB")

# 估算总大小
estimated_total_mb = (single_sample_bytes * TOTAL_SAMPLES) / (1024 * 1024)
print(f"\n>>> 预估加载 {TOTAL_SAMPLES} 条数据将占用: {estimated_total_mb:.2f} MB 的系统RAM <<<")

# 获取系统可用内存信息
try:
    import psutil
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    print(f"当前系统可用RAM: {available_gb:.2f} GB")
    if estimated_total_mb / 1024 > available_gb * 0.8: # 80% 阈值
        print("\n[!!!] 严重警告：预估内存占用超过了系统可用内存的80%！极有可能导致系统使用交换空间而“挂起”。")
    else:
        print("\n[OK] 内存检查通过：预估内存在系统可用范围内。")
except ImportError:
    print("\n请 `pip install psutil` 以自动检查系统可用内存。")