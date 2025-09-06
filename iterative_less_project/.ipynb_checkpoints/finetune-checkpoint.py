# /root/iterative_less_project/finetune.py
import argparse
import logging
import os
import torch
import trl
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
from trl import SFTTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="通用LoRA微调脚本 (SFTTrainer)")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_train_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--bf16", action='store_true', default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    args = parser.parse_args()
    set_seed(args.seed)

    logger.info(f"--- 使用 TRL SFTTrainer (v{trl.__version__}) 开始微调 ---")
    logger.info(f"实验参数: {vars(args)}")

    dataset = load_dataset("json", data_files=args.train_file, split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        save_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_first_step=True,
        data_seed=args.seed,
        optim=args.optim,
    )
    
    trainer = SFTTrainer(
        model=model, args=training_args, train_dataset=dataset,
        peft_config=lora_config,
    )
    
    trainer.train()

    logger.info(f"保存模型、分词器和优化器状态到 {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    optimizer_path = os.path.join(args.output_dir, "optimizer.pt")
    
    if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
        # --- 【核心修复】 ---
        # 直接保存优化器的 state_dict，不再手动添加 "state" 键。
        torch.save(trainer.optimizer.state_dict(), optimizer_path)
        # --- 【修复结束】 ---
        logger.info(f"优化器状态已成功保存到: {optimizer_path}")
    else:
        logger.warning("警告：在 trainer 对象上找不到优化器，无法保存其状态。")

    del model, trainer, dataset; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()