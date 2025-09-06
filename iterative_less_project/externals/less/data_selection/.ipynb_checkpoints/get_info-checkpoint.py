# /root/iterative_less_project/externals/less/data_selection/get_info.py
import argparse
import os
import torch
import logging
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader

import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

from iterative_less.progress_logger import ProgressLogger
from externals.less.data_selection.collect_grad_reps import collect_grads
from externals.less.data_selection.get_training_dataset import get_training_dataset
# --- 【核心修复】 ---
# 明确导入 get_dataloader 辅助函数
from externals.less.data_selection.get_validation_dataset import get_dataloader as get_validation_dataloader, get_dataset
# --- 【修复结束】 ---

def load_model(model_name_or_path: str, base_model_path: str, progress: ProgressLogger) -> any:
    device = "cuda"
    is_peft_adapter = os.path.exists(os.path.join(model_name_or_path, "adapter_config.json"))
    if is_peft_adapter:
        if not base_model_path: raise ValueError("Must provide --base_model_path for LoRA adapter")
        progress.log(f"Loading base model from {base_model_path} (bf16)...")
        model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map={'': device}, trust_remote_code=True, local_files_only=True)
        progress.log(f"Attaching LoRA adapter from {model_name_or_path}...")
        model = PeftModel.from_pretrained(model, model_name_or_path, is_trainable=True, local_files_only=True)
    else:
        progress.log(f"Loading full model from {model_name_or_path}...")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map={'': device}, trust_remote_code=True, local_files_only=True)
    if isinstance(model, PeftModel): model.print_trainable_parameters()
    return model

def main():
    parser = argparse.ArgumentParser(description='Script for getting gradients')
    parser.add_argument("--progress_log_file", type=str, required=True)
    parser.add_argument('--task', type=str, default=None); parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--info_type", choices=["grads"], required=True)
    parser.add_argument("--model_path", type=str, required=True); parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None); parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--output_path", type=str, required=True); parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--gradient_projection_dimension", nargs='+', type=int, default=[8192])
    parser.add_argument("--gradient_type", type=str, default="adam"); parser.add_argument("--chat_format", type=str, default="tulu")
    parser.add_argument("--use_chat_format", action='store_true', default=True); parser.add_argument("--max_length", type=int, default=1024)
    args = parser.parse_args()
    assert args.task is not None or args.train_file is not None

    progress = ProgressLogger(args.progress_log_file)
    
    try:
        progress.log("STEP 1: Preparing dataset on CPU...")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_path if args.base_model_path else args.model_path, use_fast=True, local_files_only=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

        data_iterable = None
        if args.task:
            # --- 【核心修复】 ---
            # 直接使用 get_validation_dataloader 来创建包含正确 DataCollator 的 DataLoader
            # 这可以确保 DataLoader 产生的是批次化的 Tensor，而不是 Python 列表
            dataset = get_dataset(args.task, data_dir=args.data_dir, tokenizer=tokenizer, max_length=args.max_length, use_chat_format=args.use_chat_format, chat_format=args.chat_format)
            data_iterable = get_validation_dataloader(dataset, tokenizer, batch_size=1)
            progress.log(f"已为MMLU任务创建了正确的 DataLoader (包含DataCollator)。")
            # --- 【修复结束】 ---
        else:
            progress.log("开始处理训练数据集...")
            processed_dataset_object = get_training_dataset(args.train_file, tokenizer, args.max_length, num_workers=1)
            progress.log(f"数据集处理完毕。为避免死锁，正在将 {len(processed_dataset_object)} 个样本预加载到内存中的Python列表...")
            data_iterable = [processed_dataset_object[i] for i in range(len(processed_dataset_object))]
            progress.log("数据已成功加载到内存列表，准备开始梯度计算。")
        
        progress.log("✅ Dataset preparation complete.")
        progress.log("STEP 2: Loading model to GPU...")
        model = load_model(args.model_path, args.base_model_path, progress)

        adam_param_states = None
        if args.info_type == "grads" and args.gradient_type == "adam":
            progress.log("Loading Adam optimizer state...")
            optimizer_path = os.path.join(args.model_path, "optimizer.pt")
            if not os.path.exists(optimizer_path): raise FileNotFoundError(f"Optimizer state file not found at {optimizer_path}")
            optimizer_state_dict = torch.load(optimizer_path, map_location="cpu")
            if "state" in optimizer_state_dict: adam_param_states = optimizer_state_dict["state"]
            elif all(isinstance(k, int) for k in optimizer_state_dict.keys()): adam_param_states = optimizer_state_dict
            else: raise ValueError("Could not find a valid 'state' dictionary in optimizer.pt")
        
        if args.info_type == "grads":
            progress.log("STEP 3: Starting gradient computation...")
            collect_grads(
                data_iterable=data_iterable, model=model,
                output_dir=args.output_path, 
                proj_dim=args.gradient_projection_dimension, 
                gradient_type=args.gradient_type, 
                adam_optimizer_state=adam_param_states, 
                max_samples=args.max_samples
            )
        
        progress.log("✅ All steps completed successfully!")
    except Exception as e:
        progress.log(f"FATAL ERROR: {e}")
        logger.critical("Subprocess get_info.py encountered a fatal error!", exc_info=True)
        raise

if __name__ == "__main__":
    main()