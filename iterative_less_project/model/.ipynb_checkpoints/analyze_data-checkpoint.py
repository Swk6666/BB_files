# analyze_data.py
import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CANDIDATE_POOL_PATH = 'data/candidate_pool.jsonl' 

# Hugging Face默认的截断长度，我们用它作为参考基准
TOKENIZER_MAX_LEN = 1024 

def main():
    if not pd:
        logging.error("Pandas is not installed. Please run `pip install pandas`.")
        return

    logging.info(f"--- 正在对 {CANDIDATE_POOL_PATH} 进行深度数据分析 ---")
    
    try:
        df = pd.read_json(CANDIDATE_POOL_PATH, lines=True)
    except Exception as e:
        logging.error(f"无法读取JSONL文件: {e}")
        return

    if 'text' not in df.columns:
        logging.error("文件中缺少 'text' 列。请先运行数据准备脚本。")
        return

    # 1. 检查空样本或仅包含空白的样本
    df['text_stripped'] = df['text'].str.strip()
    empty_samples = df[df['text_stripped'] == '']
    if not empty_samples.empty:
        logging.warning(f"\n[!!!] 发现了 {len(empty_samples)} 个空的或只有空白的样本！")
        logging.warning("这些样本的 original_index 如下:")
        logging.warning(empty_samples['original_index'].tolist())
    else:
        logging.info("\n[OK] 数据完整性检查：未发现空样本。")

    # 2. 分析文本长度
    df['text_length'] = df['text'].str.len()
    
    logging.info("\n--- 文本长度统计 (按字符数) ---")
    print(df['text_length'].describe())
    
    # 3. 寻找极端异常值
    # 我们定义一个“可疑”阈值，比如数倍于常规的分词器最大长度
    suspicious_threshold = TOKENIZER_MAX_LEN * 10  # 约1万个字符
    
    long_samples = df[df['text_length'] > suspicious_threshold]
    
    if not long_samples.empty:
        logging.warning(f"\n[!!!] 发现了 {len(long_samples)} 个超长样本 (长度 > {suspicious_threshold} 字符)！")
        logging.warning("这些样本可能导致VRAM耗尽或挂起。它们的详细信息如下：")
        for index, row in long_samples.iterrows():
            logging.warning(f"  - original_index: {row.get('original_index', 'N/A')}, 长度: {row['text_length']}")
    else:
        logging.info(f"\n[OK] 长度一致性检查：未发现超过 {suspicious_threshold} 字符的极端异常样本。")

if __name__ == "__main__":
    main()