# /root/iterative_less_project/run_entropy_scoring.py
import argparse
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="独立的、健壮的熵评分脚本。")
    parser.add_argument("--mt5_model_path", type=str, required=True)
    parser.add_argument("--input_df_path", type=str, required=True)
    parser.add_argument("--output_scores_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    df_to_score = pd.read_json(args.input_df_path, lines=True)
    logger.info(f"开始使用 {args.mt5_model_path} 计算 {len(df_to_score)} 条数据的预测熵...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.mt5_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.mt5_model_path)
    model.eval()
    
    results = []
    for i in tqdm(range(0, len(df_to_score), args.batch_size), desc="Entropy Scoring"):
        batch_df = df_to_score.iloc[i:i+args.batch_size]
        original_indices = batch_df['original_index'].tolist()
        
        if 'masked_text' not in batch_df.columns:
            raise ValueError("输入文件中缺少 'masked_text' 列！")
        
        input_texts = [row['masked_text'].split(" Target: ")[0] for _, row in batch_df.iterrows()]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
        
        try:
            outputs = model.generate(**inputs, max_length=256, return_dict_in_generate=True, output_scores=True)
            
            batch_entropies = [[] for _ in range(len(batch_df))]
            if outputs.scores:
                for step_logits in outputs.scores:
                    probs = F.softmax(step_logits, dim=-1)
                    log_probs = F.log_softmax(step_logits, dim=-1)
                    step_entropies = -torch.sum(probs * log_probs, dim=-1)
                    step_entropies = torch.nan_to_num(step_entropies, nan=0.0) # 核心保护
                    
                    for j, entropy in enumerate(step_entropies):
                        batch_entropies[j].append(entropy.item())

            for j in range(len(batch_df)):
                avg_entropy = np.mean(batch_entropies[j]) if batch_entropies[j] else 0.0
                results.append({"original_index": original_indices[j], "imputation_score": avg_entropy})
        
        except Exception as e:
            logger.error(f"处理批次 {i//args.batch_size} 时发生错误: {e}")
            for idx in original_indices:
                results.append({"original_index": idx, "imputation_score": -1.0})

    scores_df = pd.DataFrame(results)
    scores_df.to_json(args.output_scores_path, orient='records', lines=True)
    logger.info(f"✅ 熵评分完成，结果已保存到: {args.output_scores_path}")

if __name__ == "__main__":
    main()