# /root/iterative_less_project/prepare_data_final.py
import json
import random
import os
import logging
import re
import numpy as np
from tqdm import tqdm

# --- 配置区 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

class Config:
    RANDOM_SEED = 42
    BASE_DATA_DIR = "data"
    RAW_DATA_PATHS = {
        "flan_v2": os.path.join(BASE_DATA_DIR, "raw_data/flan_v2_data.jsonl"),
        "cot": os.path.join(BASE_DATA_DIR, "raw_data/cot_data.jsonl"),
        "dolly": os.path.join(BASE_DATA_DIR, "raw_data/dolly_data.jsonl"),
        "oasst1": os.path.join(BASE_DATA_DIR, "raw_data/oasst1_data.jsonl"),
    }
    DATA_SOURCES_TO_POOL = ["flan_v2", "cot", "dolly", "oasst1"]
    
    NUM_WARMUP_SAMPLES = 300
    NUM_CANDIDATE_SAMPLES = 300

    WARMUP_POOL_FILENAME = os.path.join(BASE_DATA_DIR, "warmup_pool.jsonl")
    CANDIDATE_POOL_FILENAME = os.path.join(BASE_DATA_DIR, "candidate_pool.jsonl")

    NOISE_DENSITY = 0.25
    MEAN_NOISE_SPAN_LENGTH = 3.0

config = Config()
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

# --- 辅助函数 ---
def format_chat_messages_to_text(messages):
    """将 messages 列表转换为单个文本字符串。"""
    eos_token = "</s>"
    text = ""
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content', '')
        if role == 'system':
            text += f"<|system|>\n{content}\n"
        elif role == 'user':
            text += f"<|user|>\n{content}\n"
        elif role == 'assistant':
            text += f"<|assistant|>\n{content}{eos_token}\n"
    return text.strip()

def apply_t5_span_corruption(text, noise_density, mean_noise_span_length):
    """实现T5风格的“片段腐蚀”任务，失败时返回None。"""
    tokens = text.split()
    num_tokens = len(tokens)
    num_to_noise = int(round(num_tokens * noise_density))
    if num_to_noise == 0:
        return None # 返回None表示未成功遮蔽

    num_noise_spans = int(round(num_to_noise / mean_noise_span_length))
    if num_noise_spans == 0:
        return None # 返回None表示未成功遮蔽

    lengths = np.random.poisson(mean_noise_span_length, num_noise_spans)
    lengths = np.clip(lengths, 1, num_tokens)
    
    indices_to_start_spans = sorted(np.random.choice(num_tokens, size=num_noise_spans, replace=False))
    
    masked_spans, last_covered_idx = [], -1
    for i, start_index in enumerate(indices_to_start_spans):
        if start_index <= last_covered_idx: continue
        end_index = min(start_index + int(lengths[i]), num_tokens)
        masked_spans.append((start_index, end_index))
        last_covered_idx = end_index - 1

    if not masked_spans:
        return None # 返回None表示未成功遮蔽

    input_parts, target_parts, last_unmasked_idx, sentinel_id = [], [], 0, 0
    for start, end in masked_spans:
        if start > last_unmasked_idx: input_parts.append(" ".join(tokens[last_unmasked_idx:start]))
        sentinel_token = f"<extra_id_{sentinel_id}>"
        input_parts.append(sentinel_token)
        target_parts.extend([sentinel_token, " ".join(tokens[start:end])])
        last_unmasked_idx = end
        sentinel_id += 1
        if sentinel_id >= 100: break

    if last_unmasked_idx < num_tokens: input_parts.append(" ".join(tokens[last_unmasked_idx:]))
    target_parts.append(f"<extra_id_{sentinel_id}>")
    
    final_input = re.sub(r'\s+', ' ', " ".join(filter(None, input_parts)).strip())
    final_target = " ".join(filter(None, target_parts)).strip()
    return f"Input: {final_input} Target: {final_target}"

# --- 主逻辑 ---
def main():
    os.makedirs(config.BASE_DATA_DIR, exist_ok=True)
    logging.info("--- 结构化数据准备开始 ---")

    logging.info("[步骤 1/3] 聚合所有源数据...")
    master_pool = []
    for name in config.DATA_SOURCES_TO_POOL:
        source_path = config.RAW_DATA_PATHS.get(name)
        if not os.path.exists(source_path):
            logging.error(f"错误：源数据文件未找到: {name} -> {source_path}")
            continue
        with open(source_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'messages' in data and data['messages']:
                        data['source_dataset'] = name
                        master_pool.append(data)
                except json.JSONDecodeError:
                    continue
    
    logging.info(f"\n[步骤 2/3] 随机打乱 {len(master_pool)} 条数据...")
    random.shuffle(master_pool)
    
    logging.info("\n[步骤 3/3] 处理并生成最终数据集（将过滤未成功遮蔽的数据）...")
    processed_data = []
    filtered_count = 0
    
    with tqdm(total=config.NUM_WARMUP_SAMPLES + config.NUM_CANDIDATE_SAMPLES, desc="处理和筛选数据") as pbar:
        for i, data_point in enumerate(master_pool):
            if len(processed_data) >= config.NUM_WARMUP_SAMPLES + config.NUM_CANDIDATE_SAMPLES:
                break

            full_text = format_chat_messages_to_text(data_point['messages'])
            conversation_text = " ".join([msg['content'] for msg in data_point['messages'] if msg['role'] in ['user', 'assistant']]).strip()
            if not conversation_text:
                filtered_count += 1
                continue
            
            masked_text_combined = apply_t5_span_corruption(
                conversation_text, 
                config.NOISE_DENSITY, 
                config.MEAN_NOISE_SPAN_LENGTH
            )
            
            if not masked_text_combined:
                filtered_count += 1
                continue

            # 强制确保所有字段都是字符串类型，防止JSON解析错误
            processed_data.append({
                "original_index": i,
                "full_text": str(full_text),
                "masked_text": str(masked_text_combined),
                "source_dataset": str(data_point.get('source_dataset', 'unknown'))
            })
            pbar.update(1)

    if len(processed_data) < config.NUM_WARMUP_SAMPLES + config.NUM_CANDIDATE_SAMPLES:
        logging.error(f"数据不足！聚合了 {len(master_pool)} 条，但只有 {len(processed_data)} 条数据成功处理，少于所需的 {config.NUM_WARMUP_SAMPLES + config.NUM_CANDIDATE_SAMPLES} 条。")
        return
        
    logging.info(f"数据处理完成。总共筛选掉了 {filtered_count} 条过短或无法遮蔽的数据。")

    # 切分预热集和候选池
    warmup_pool = processed_data[:config.NUM_WARMUP_SAMPLES]
    candidate_pool = processed_data[config.NUM_WARMUP_SAMPLES:]

    # 保存文件
    with open(config.WARMUP_POOL_FILENAME, 'w', encoding='utf-8') as f:
        for item in warmup_pool:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logging.info(f"预热数据集创建完成，已保存 {len(warmup_pool)} 条到 {config.WARMUP_POOL_FILENAME}")

    with open(config.CANDIDATE_POOL_FILENAME, 'w', encoding='utf-8') as f:
        for item in candidate_pool:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logging.info(f"候选数据池创建完成，已保存 {len(candidate_pool)} 条到 {config.CANDIDATE_POOL_FILENAME}")

    logging.info("\n--- 所有数据准备就绪！ ---")

if __name__ == "__main__":
    main()