# /root/iterative_less_project/run_imputation.py
import argparse
import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="独立的文本插补脚本。")
    parser.add_argument("--mt5_model_path", type=str, required=True)
    parser.add_argument("--input_df_path", type=str, required=True)
    parser.add_argument("--output_df_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    df_to_impute = pd.read_json(args.input_df_path, lines=True)
    logger.info(f"开始使用 {args.mt5_model_path} 对 {len(df_to_impute)} 条数据进行插补...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.mt5_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.mt5_model_path)
    model.eval()
    
    imputed_texts = []
    for i in tqdm(range(0, len(df_to_impute), args.batch_size), desc="Imputing Text"):
        batch_df = df_to_impute.iloc[i:i+args.batch_size]
        input_texts = [row['masked_text'].split(" Target: ")[0] for _, row in batch_df.iterrows()]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
        generated_ids = model.generate(**inputs, max_length=512, num_beams=1, early_stopping=True)
        batch_imputed_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        imputed_texts.extend(batch_imputed_texts)
        
    df_imputed = df_to_impute.copy()
    df_imputed['imputed_text'] = imputed_texts
    
    df_imputed.to_json(args.output_df_path, orient='records', lines=True, force_ascii=False)
    logger.info(f"插补完成，结果已保存到: {args.output_df_path}")

if __name__ == "__main__":
    main()