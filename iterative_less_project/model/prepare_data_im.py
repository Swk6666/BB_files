# /root/iterative_less_project/prepare_data.py
import json
import random
import os
import logging
import re
import string
# 导入numpy用于科学计算，如此处的泊松分布
import numpy as np
# 导入tqdm用于显示美观的进度条
from tqdm import tqdm
# 从transformers库导入AutoTokenizer，用于加载预训练模型的分词器
from transformers import AutoTokenizer

# --- 配置区 ---
# 定义一个配置类，用于存放全局常量
class Config:
    RANDOM_SEED = 42

# 实例化配置
config = Config()

# 设置日志记录，方便观察脚本运行状态和调试
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - [%(filename)s:%(lineno)d] - %(message)s')

# ==============================================================================
# ---                           配置区域                                   ---
# ==============================================================================
# 所有数据文件的基础目录 (位于系统盘)
BASE_DATA_DIR = "data"
# 指定用于处理文本（如此处判断长度）的分词器名称
TOKENIZER_FOR_PROCESSING = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
# 定义所有模型处理的最大Token长度，这是一个全局约束
MAX_TOKEN_LENGTH = 512

# 原始数据文件的路径
RAW_DATA_PATHS = {
    "flan_v2": os.path.join(BASE_DATA_DIR, "raw_data/flan_v2_data.jsonl"),
    "cot": os.path.join(BASE_DATA_DIR, "raw_data/cot_data.jsonl"),
    "dolly": os.path.join(BASE_DATA_DIR, "raw_data/dolly_data.jsonl"),
    "oasst1": os.path.join(BASE_DATA_DIR, "raw_data/oasst1_data.jsonl"),
}
# 定义要使用的数据源
DATA_SOURCES_TO_POOL = ["flan_v2", "cot", "dolly", "oasst1"]
# 定义为预热（warmup）阶段准备的样本数量
NUM_WARMUP_SAMPLES = 3000
# 定义最终生成的候选数据池的样本数量
NUM_CANDIDATE_SAMPLES = 5000

# 定义输出文件的完整路径
CANDIDATE_FILENAME = os.path.join(BASE_DATA_DIR, "candidate_pool_imputation_spans.jsonl")
WARMUP_FILENAME = os.path.join(BASE_DATA_DIR, "warmup_data_for_sft.jsonl")
IMPUTATION_WARMUP_FILENAME = os.path.join(BASE_DATA_DIR, "warmup_data_for_imputation_spans.jsonl")

# Span Corruption (片段腐蚀/插补) 任务的参数
NOISE_DENSITY = 0.3  # 遮蔽文本中15%的token
MEAN_NOISE_SPAN_LENGTH = 5.0 # 被遮蔽片段的平均长度为3个token
# ==============================================================================

def format_chat_messages_to_text(messages):
    """
    将结构化的对话`messages`列表转换为单个文本字符串。
    这是为了适配SFTTrainer等需要纯文本输入的训练框架。
    """
    eos_token = "</s>"  # 句尾符
    text = ""
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content', '')
        # 根据角色添加特定的标记，构建对话格式
        if role == 'system':
            text += f"<|system|>\n{content}\n"
        elif role == 'user':
            text += f"<|user|>\n{content}\n"
        elif role == 'assistant':
            text += f"<|assistant|>\n{content}{eos_token}\n"
    return text

def detect_language_and_split(text):
    """
    一个简单的函数，用于将文本分割成“单词”列表，为后续的随机遮蔽做准备。
    """
    if not text or not text.strip():
        return []
    # 通过在常见标点符号前后添加空格，然后按空格分割，实现简单的分词
    text = re.sub(r'([,.!?])', r' \1 ', text)
    tokens = text.split()
    return [t for t in tokens if t.strip()]

def apply_t5_span_corruption(text, noise_density, mean_noise_span_length):
    """
    实现T5风格的“片段腐蚀”任务，生成用于插补模型训练的数据。
    """
    # 步骤1：将文本分割成词元
    tokens = detect_language_and_split(text)
    if not tokens: return text, ""

    num_tokens = len(tokens)
    # 计算总共需要遮蔽的词元数量
    num_to_noise = int(round(num_tokens * noise_density))
    if num_to_noise == 0: return text, ""

    # 计算需要生成的遮蔽片段（span）的数量
    num_noise_spans = min(int(round(num_to_noise / mean_noise_span_length)) or 1, 99)
    # 使用泊松分布为每个片段生成一个随机的长度，使其在平均长度附近波动
    lengths = np.random.poisson(mean_noise_span_length, num_noise_spans)
    
    # 调整生成的长度，确保总长不超过计划遮蔽的总词元数
    if np.sum(lengths) > num_to_noise:
        while np.sum(lengths) > num_to_noise:
            if np.any(lengths > 1):
                lengths[np.argmax(lengths)] -= 1
            else: break

    # 随机选择遮蔽片段的起始位置
    indices_to_start_spans = sorted(np.random.choice(num_tokens, size=num_noise_spans, replace=False))
    
    # 确定最终的遮蔽片段，并处理重叠情况
    masked_spans, last_covered_idx = [], -1
    for i, start_index in enumerate(indices_to_start_spans):
        if start_index <= last_covered_idx: continue
        end_index = min(start_index + lengths[i], num_tokens)
        masked_spans.append((start_index, end_index))
        last_covered_idx = end_index - 1

    if not masked_spans: return text, ""

    # 构建模型的输入文本（带<extra_id_..>标记）和目标文本
    input_parts, target_parts, last_unmasked_idx, sentinel_id = [], [], 0, 0
    for start, end in masked_spans:
        if start > last_unmasked_idx: input_parts.append(" ".join(tokens[last_unmasked_idx:start]))
        sentinel_token = f"<extra_id_{sentinel_id}>"
        input_parts.append(sentinel_token)
        target_parts.extend([sentinel_token, " ".join(tokens[start:end])])
        last_unmasked_idx = end
        sentinel_id += 1

    if last_unmasked_idx < num_tokens: input_parts.append(" ".join(tokens[last_unmasked_idx:]))
    target_parts.append(f"<extra_id_{sentinel_id}>")
    
    # 拼接并清理最终的文本
    return " ".join(filter(None, input_parts)).strip(), " ".join(filter(None, target_parts)).strip()

def process_data_point_for_imputation(data_point, data_index, tokenizer):
    """
    处理单个数据点，为其生成插补任务所需的输入和目标。
    实现了核心逻辑：如果文本超长，则只在其前512个token内进行遮蔽。
    """
    try:
        messages = data_point.get('messages', [])
        # 将用户和助手的对话合并成一个长字符串，作为插补任务的原始内容
        full_conversation_text = " ".join([msg['content'] for msg in messages if msg['role'] in ['user', 'assistant']]).strip()
        
        if not full_conversation_text or len(full_conversation_text) < 20:
            return None
        
        # 保留一份完整的、用于主模型SFT训练的文本
        full_text_sft = format_chat_messages_to_text(messages)
        
        # --- 核心逻辑：判断长度并截断 ---
        # 首先用分词器计算原始文本的真实token数量
        token_ids = tokenizer(full_conversation_text, truncation=False)['input_ids']
        
        if len(token_ids) > MAX_TOKEN_LENGTH:
            # 1. 如果文本超长，则只截取前MAX_TOKEN_LENGTH个token ID
            truncated_token_ids = token_ids[:MAX_TOKEN_LENGTH]
            # 2. 将这些截断后的ID解码回字符串，作为我们进行遮蔽操作的基础文本
            text_to_mask = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
            logging.debug(f"样本 {data_index} 超长 ({len(token_ids)} tokens)，已截断至前 {MAX_TOKEN_LENGTH} tokens 进行遮蔽。")
        else:
            # 3. 如果文本长度在限制内，则直接使用全文进行遮蔽
            text_to_mask = full_conversation_text
        # --- 逻辑结束 ---

        # 设置随机种子以保证结果可复现
        random.seed(data_index + config.RANDOM_SEED + hash(full_conversation_text[:50]))
        # 对我们准备好的（可能被截断的）文本执行片段腐蚀操作
        input_text, target_text = apply_t5_span_corruption(text_to_mask, NOISE_DENSITY, MEAN_NOISE_SPAN_LENGTH)

        if not target_text:
            return None

        # 返回一个包含所有需要信息的字典
        return {
            "input_text": input_text,               # 给插补模型的输入 (带遮蔽标记)
            "target_text": target_text,             # 给插补模型的目标 (被遮蔽的内容)
            "full_text_sft": full_text_sft,         # 给主模型SFT训练的完整原文
            "original_full_text": full_conversation_text, # 原始的对话原文，用于分析和对比
            "source_dataset": data_point.get('source_dataset', 'unknown'),
        }
    except Exception as e:
        logging.error(f"数据点{data_index}处理时发生严重错误: {e}")
        return None

def main():
    """主执行函数"""
    # 确保数据目录存在
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    logging.info(f"--- 数据准备开始 (输出到: {BASE_DATA_DIR}) ---")
    
    # 只需要在主函数中实例化一次分词器，然后传递给需要的函数，提高效率
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_FOR_PROCESSING)

    # 步骤 1: 聚合所有源数据到一个大的列表中
    logging.info(f"[步骤 1/4] 聚合所有源数据...")
    master_pool = []
    for name in DATA_SOURCES_TO_POOL:
        source_path = RAW_DATA_PATHS.get(name)
        if not os.path.exists(source_path):
            logging.error(f"错误：源数据文件未找到: {name} -> {source_path}")
            return
        try:
            with open(source_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if 'messages' in data:
                            data['source_dataset'] = name
                            master_pool.append(data)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logging.error(f"加载文件 {source_path} 时出错: {e}")
            return
            
    # 检查聚合后的数据是否足够
    total_required = NUM_WARMUP_SAMPLES + NUM_CANDIDATE_SAMPLES
    if len(master_pool) < total_required:
        logging.error(f"数据不足！需要 {total_required} 条，但只有 {len(master_pool)} 条。请准备更多数据。")
        return

    # 步骤 2: 随机打乱聚合后的数据池
    logging.info(f"\n[步骤 2/4] 随机打乱 {len(master_pool)} 条数据...")
    random.seed(config.RANDOM_SEED)
    random.shuffle(master_pool)

    # 步骤 3: 切分并处理预热数据
    logging.info(f"\n[步骤 3/4] 处理 {NUM_WARMUP_SAMPLES} 条预热数据...")
    warmup_data_raw = master_pool[:NUM_WARMUP_SAMPLES]

    # 生成用于主模型（如TinyLlama）SFT微调的预热文件
    with open(WARMUP_FILENAME, 'w', encoding='utf-8') as f:
        for data in warmup_data_raw:
            f.write(json.dumps({"text": format_chat_messages_to_text(data['messages'])}, ensure_ascii=False) + '\n')
    logging.info(f"为SFT主任务准备的预热数据已保存到: {WARMUP_FILENAME}")

    # 生成用于插补模型（mT5）微调的预热文件
    with open(IMPUTATION_WARMUP_FILENAME, 'w', encoding='utf-8') as f:
        count = 0
        for i, data in enumerate(warmup_data_raw):
            # 将tokenizer实例传入，避免重复加载
            processed_data = process_data_point_for_imputation(data, i, tokenizer)
            if processed_data:
                f.write(json.dumps(processed_data, ensure_ascii=False) + '\n')
                count += 1
    logging.info(f"为插补模型准备的预热数据已处理并保存 {count} 条到: {IMPUTATION_WARMUP_FILENAME}")

    # 步骤 4: 切分并处理候选数据池
    logging.info(f"\n[步骤 4/4] 应用 Span Corruption 到候选数据池...")
    candidate_pool_raw = master_pool[NUM_WARMUP_SAMPLES : total_required]
    
    processed_candidates = []
    # 使用tqdm显示处理进度
    for i, data in enumerate(tqdm(candidate_pool_raw, desc="处理候选数据池")):
        # 将tokenizer实例传入，并使用不同的随机种子以增加多样性
        processed_data = process_data_point_for_imputation(data, i + NUM_WARMUP_SAMPLES, tokenizer)
        if processed_data:
            processed_data["original_index"] = i
            processed_candidates.append(processed_data)

    # 将处理好的候选数据写入文件
    with open(CANDIDATE_FILENAME, 'w', encoding='utf-8') as f:
        for data in processed_candidates:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    logging.info(f"\n候选数据池创建完成，已处理并保存 {len(processed_candidates)} 条到 {CANDIDATE_FILENAME}")
    logging.info("\n--- 所有数据准备就绪！ ---")

# Python脚本的入口点
if __name__ == "__main__":
    main()