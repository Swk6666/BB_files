# /root/iterative_less_project/externals/less/data_selection/get_validation_dataset.py
import json
import os
from typing import List, Tuple

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedTokenizerBase

B_INST, E_INST = "[INST]", "[/INST]"

def tokenize(tokenizer: PreTrainedTokenizerBase,
             query: str,
             completion: str,
             max_length: int,
             print_ex: bool = False) -> Tuple[List[int], List[int], List[int]]:
    """
    Formats a chat conversation into input tensors for a transformer model.
    """
    full_prompt = query + completion
    if print_ex:
        print(f"******** Example starts ********\n{full_prompt}\n******** Example ends ********")

    query_input_ids = tokenizer.encode(query, add_special_tokens=False)
    full_input_ids = tokenizer.encode(full_prompt, add_special_tokens=False, max_length=max_length, truncation=True)
    
    labels = list(full_input_ids)
    labels[:len(query_input_ids)] = [-100] * len(query_input_ids)
    attention_mask = [1] * len(full_input_ids)

    return full_input_ids, labels, attention_mask

def get_mmlu_dataset(data_dir: str,
                     tokenizer: PreTrainedTokenizerBase,
                     max_length: int,
                     use_chat_format=True,
                     chat_format="tulu",
                     **kwargs):
    """
    Loads MMLU dev set samples to build the validation set D_val.
    """
    # --- 【核心修复】 ---
    # 移除所有对不存在的 "eval" 目录的引用。
    # 我们从 config.py 接收的 data_dir 是 '.../data'，
    # 所以 mmlu_data_dir 应该是 '.../data/mmlu'。
    mmlu_data_dir = os.path.join(data_dir, "mmlu")
    # --- 【修复结束】 ---
    
    try:
        subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(mmlu_data_dir, "test")) if "_test.csv" in f])
    except FileNotFoundError:
        print(f"错误: MMLU 'test' 目录未在 {os.path.join(mmlu_data_dir, 'test')} 找到，无法获取科目列表。")
        print(f"请确认您的目录结构是 '.../data/mmlu/test' 和 '.../data/mmlu/dev'")
        raise
    
    print(f"从 {len(subjects)} 个MMLU科目中加载DEV集来构建D_val...")

    def format_subject(subject):
        return " ".join(subject.split("_"))

    def format_example(df, idx, include_answer=True):
        choices = ["A", "B", "C", "D"]
        prompt = str(df.iloc[idx, 0])
        k = df.shape[1] - 2
        for j in range(k):
            prompt += f"\n{choices[j]}. {df.iloc[idx, j + 1]}"
        prompt += "\nAnswer:"
        if include_answer:
            prompt += " " + str(df.iloc[idx, k + 1])
        return prompt

    def gen_prompt_from_dev(dev_df, subject, target_idx):
        prompt = f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
        prompt += format_example(dev_df, target_idx, include_answer=False)
        return prompt

    k = 5
    dataset = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for subject in subjects:
        try:
            dev_df = pd.read_csv(os.path.join(mmlu_data_dir, "dev", f"{subject}_dev.csv"), header=None, encoding='utf-8', encoding_errors='ignore')
        except FileNotFoundError:
            print(f"警告: 未找到科目 '{subject}' 的dev集，已跳过。")
            continue
            
        num_samples_to_use = min(k, len(dev_df))
        if num_samples_to_use == 0: continue
            
        dev_df_subset = dev_df.head(num_samples_to_use)

        for i in range(len(dev_df_subset)):
            prompt = gen_prompt_from_dev(dev_df_subset, subject, i)
            answer = " " + str(dev_df_subset.iloc[i, dev_df_subset.shape[1] - 1])
            
            if use_chat_format:
                if chat_format == "tulu":
                    prompt = f"<|user|>\n{prompt}\n<|assistant|>\nThe answer is:"
                else:
                    prompt = f"<s>{B_INST} {prompt.strip()} {E_INST} The answer is:"
            
            full_input_ids, labels, attention_mask = tokenize(
                tokenizer, prompt, answer, max_length, print_ex=(i == 0 and subject == subjects[0])
            )
            dataset["input_ids"].append(full_input_ids)
            dataset["labels"].append(labels)
            dataset["attention_mask"].append(attention_mask)

    return Dataset.from_dict(dataset)

def get_dataset(task, **kwargs):
    if task == "mmlu":
        return get_mmlu_dataset(**kwargs)
    else:
        raise ValueError(f"任务 '{task}' 的验证集加载逻辑未实现。")

def get_dataloader(dataset, tokenizer, batch_size=1):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest") 
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    print(f"为梯度计算准备了 {len(dataset)} 个验证样本。")
    return dataloader