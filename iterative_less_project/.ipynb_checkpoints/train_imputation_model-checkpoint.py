# /root/iterative_less_project/train_imputation_model.py
import logging
import os
import torch
import argparse
from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# ---                           配置区域                                   ---
# ==============================================================================
# 基础模型名称，将通过Hugging Face的缓存机制自动管理
MODEL_NAME = "google/mt5-base"

# --- 【核心修改 1】性能优化 (适配新服务器) ---
EPOCHS = 2
BATCH_SIZE = 4  # 增大设备批次大小
GRADIENT_ACCUMULATION_STEPS = 32 # 对应减小梯度累积，保持有效批大小128
LEARNING_RATE = 3e-4
LORA_CONFIG = {
    "r": 32, "lora_alpha": 64, "lora_dropout": 0.05, "bias": "none",
    "task_type": TaskType.SEQ_2_SEQ_LM, "target_modules": ["q", "v"],
}
# ==============================================================================

# --- 【核心修改 2】移除硬编码路径，依赖环境变量 ---
# 移除了 check_and_download_model_if_needed 函数，因为设置了HF_HOME后，
# transformers库会自动处理下载和缓存到数据盘，代码更简洁、更标准。

def preprocess_function(batch, tokenizer):
    """
    核心数据预处理函数。将 "masked_text" 列拆分为输入和目标。
    """
    inputs = []
    targets = []
    # 确保处理的列名是 'masked_text'
    for item in batch["masked_text"]:
        if " Target: " in item:
            parts = item.split(" Target: ", 1)
            inputs.append(parts[0])
            targets.append(parts[1])
        else:
            inputs.append(item)
            targets.append("")
            logger.warning(f"数据格式警告：输入 '{item[:50]}...' 中未找到 ' Target: ' 分隔符。")

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(text_target=targets, max_length=256, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="训练文件的路径")
    parser.add_argument("--output_dir", type=str, required=True, help="模型输出和检查点的目录")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    torch.cuda.empty_cache()
    
    # transformers 会自动使用 HF_HOME 环境变量指定的路径
    logger.info(f"从Hugging Face Hub或本地缓存 ({os.getenv('HF_HOME')}) 加载tokenizer: {MODEL_NAME}")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

    logger.info(f"从Hugging Face Hub或本地缓存 ({os.getenv('HF_HOME')}) 加载模型: {MODEL_NAME}")
    model = MT5ForConditionalGeneration.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    logger.info(f"加载并处理数据集: {args.train_file}")
    # dataset cache会自动使用 HF_DATASETS_CACHE 环境变量
    raw_dataset = load_dataset(
        "json", 
        data_files=args.train_file, 
        split="train"
    )
    
    tokenized_dataset = raw_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True, 
        num_proc=max(os.cpu_count() // 2, 1), # 使用多进程加速数据处理
        remove_columns=raw_dataset.column_names
    )

    logger.info("应用LoRA配置...")
    peft_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        report_to="none",
        optim="adamw_torch",
        dataloader_num_workers=4, # 增加数据加载器的工作进程
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    logger.info("开始训练...")
    trainer.train()
    
    trainer.save_model(args.output_dir)
    # tokenizer 也应保存到模型目录
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"训练完成！模型已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()