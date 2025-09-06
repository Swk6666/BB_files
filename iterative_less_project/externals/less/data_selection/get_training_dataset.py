# /root/iterative_less_project/externals/less/data_selection/get_training_dataset.py
import contextlib
from functools import partial
from typing import List, Union
import numpy as np
import torch
from datasets import load_dataset
import os
import logging # <-- 新增导入

logger = logging.getLogger(__name__) # <-- 新增

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def get_training_dataset(train_files: List[str], tokenizer, max_seq_length, sample_percentage=1.0, seed=0, num_workers: int = None):
    raw_datasets = load_raw_dataset(train_files, sample_percentage=sample_percentage, seed=seed)
    
    # --- 【核心修复】 ---
    # 在梯度计算的子进程中，我们必须强制使用单进程进行数据处理，以避免嵌套多进程死锁。
    # 无论上层传入什么num_workers参数，这里都将其覆盖为1。
    processing_num_workers = 1
    logger.warning(f"为避免在子进程中发生死锁，数据处理进程数被强制设置为 {processing_num_workers}。")
    # --- 【修复结束】 ---

    lm_datasets = encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=processing_num_workers)
    
    return lm_datasets

def load_raw_dataset(train_files: Union[List[str], str], sample_size=None, sample_percentage=1.0, seed=0):
    if isinstance(train_files, str):
        train_files = [train_files]
    processed_datasets = load_dataset("json", data_files=train_files, split="train")
    
    if sample_size is None and sample_percentage < 1.0:
        sample_size = int(len(processed_datasets) * sample_percentage)
    
    if sample_size is not None and sample_size < len(processed_datasets):
        with temp_seed(seed):
            index = np.random.permutation(len(processed_datasets))[:sample_size]
        sampled_dataset = processed_datasets.select(index)
        return sampled_dataset
    return processed_datasets

def encode_data(raw_datasets, tokenizer, max_seq_length, processing_num_workers=1, overwrite_cache=False):
    if "input_ids" in raw_datasets.features:
        return raw_datasets
    
    if "text" not in raw_datasets.column_names:
        raise ValueError("Input data file must contain a 'text' column.")
    
    encode_function = partial(encode_with_text_format, tokenizer=tokenizer, max_seq_length=max_seq_length)
    
    # --- 【诊断性检查】 ---
    logger.info(f"即将开始Tokenizing数据（使用 {processing_num_workers} 个进程）。如果程序在此处长时间无响应，说明发生了死锁。")
    # --- 【检查结束】 ---
    
    lm_datasets = raw_datasets.map(
        encode_function,
        batched=False,
        num_proc=processing_num_workers,
        load_from_cache_file=not overwrite_cache,
        remove_columns=[col for col in raw_datasets.column_names if col not in ['input_ids', 'labels', 'attention_mask']],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_datasets.set_format(type="pt")
    logger.info("Tokenization和数据格式化已成功完成。")
    return lm_datasets

def encode_with_text_format(example, tokenizer, max_seq_length):
    text = example['text']
    tokenized_example = tokenizer(text, max_length=max_seq_length, truncation=True, add_special_tokens=False)
    
    input_ids = tokenized_example['input_ids']
    labels = list(input_ids)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': tokenized_example['attention_mask'],
    }