# /root/iterative_less_project/prepare_data_less.py
import json
import random
import os
import logging
from iterative_less import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

# --- 配置区 ---
RAW_DATA_PATHS = {
    "flan_v2": "data/raw_data/flan_v2_data.jsonl",
    "cot": "data/raw_data/cot_data.jsonl",
    "dolly": "data/raw_data/dolly_data.jsonl",
    "oasst1": "data/raw_data/oasst1_data.jsonl",
}
DATA_SOURCES_TO_POOL = ["flan_v2", "cot", "dolly", "oasst1"]
NUM_WARMUP_SAMPLES = 1000
NUM_CANDIDATE_SAMPLES = 20000
OUTPUT_DIR = "data"
CANDIDATE_FILENAME = "candidate_pool.jsonl"
WARMUP_FILENAME = "warmup_pool.jsonl"
# --- 【新增】安全长度阈值 (字符数)，20000个字符是一个非常安全的上限 ---
MAX_SAFE_LENGTH_CHARS = 20000 

def format_chat_messages_to_text(messages):
    eos_token = "</s>"
    text = ""
    for msg in messages:
        if msg.get('role') == 'system':
            text += f"<|system|>\n{msg['content']}\n"
        elif msg.get('role') == 'user':
            text += f"<|user|>\n{msg['content']}\n"
        elif msg.get('role') == 'assistant':
            text += f"<|assistant|>\n{msg['content']}{eos_token}\n"
    return text

def main():
    candidate_output_path = os.path.join(OUTPUT_DIR, CANDIDATE_FILENAME)
    warmup_output_path = os.path.join(OUTPUT_DIR, WARMUP_FILENAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    logging.info("--- 开始大规模数据准备 (已加入安全过滤器) ---")

    logging.info(f"[步骤 1/4] 聚合所有源数据...")
    master_pool = []
    # (省略了数据加载的详细代码，保持不变)
    for name in DATA_SOURCES_TO_POOL:
        # ...
        with open(RAW_DATA_PATHS.get(name), 'r', encoding='utf-8') as f:
            for line in f:
                try: master_pool.append(json.loads(line))
                except: continue
            
    total_required = NUM_WARMUP_SAMPLES + NUM_CANDIDATE_SAMPLES
    
    logging.info(f"\n[步骤 2/4] 随机打乱 {len(master_pool)} 条数据...")
    random.seed(config.RANDOM_SEED)
    random.shuffle(master_pool)

    logging.info(f"\n[步骤 3/4] 过滤数据并创建干净的数据池...")
    clean_pool = []
    empty_count = 0
    long_count = 0
    original_index_counter = 0

    for data in master_pool:
        # 确保 'messages' 键存在
        if 'messages' not in data: continue
            
        formatted_text = format_chat_messages_to_text(data['messages'])
        
        # 过滤器1: 检查空文本
        if not formatted_text.strip():
            empty_count += 1
            continue
            
        # 过滤器2: 检查超长文本
        if len(formatted_text) > MAX_SAFE_LENGTH_CHARS:
            long_count += 1
            continue
            
        clean_pool.append({
            "text": formatted_text,
            "original_index": original_index_counter
        })
        original_index_counter += 1

    logging.info(f"过滤完成。跳过了 {empty_count} 个空样本和 {long_count} 个超长样本。")
    logging.info(f"干净数据池大小: {len(clean_pool)}")
    
    if len(clean_pool) < total_required:
        logging.error(f"数据不足！过滤后需要 {total_required} 条，但只有 {len(clean_pool)} 条。请检查源数据或降低请求的样本数。")
        return

    logging.info(f"\n[步骤 4/4] 从干净数据池中切分预热集和候选集...")
    warmup_data = clean_pool[:NUM_WARMUP_SAMPLES]
    candidate_data = clean_pool[NUM_WARMUP_SAMPLES : total_required]

    with open(warmup_output_path, 'w', encoding='utf-8') as f:
        for item in warmup_data:
            f.write(json.dumps({"text": item["text"]}, ensure_ascii=False) + '\n')
    logging.info(f"预热数据创建完成，已保存到 {warmup_output_path}")

    with open(candidate_output_path, 'w', encoding='utf-8') as f:
        for item in candidate_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logging.info(f"候选数据池创建完成，已保存到 {candidate_output_path}")

    logging.info("\n--- 所有数据准备就绪！ ---")

if __name__ == "__main__":
    main()