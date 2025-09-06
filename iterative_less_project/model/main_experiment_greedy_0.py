# /root/iterative_less_project/main_experiment_greedy.py
import os
import json
import logging
import time
import shutil
import random
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch.nn.functional as F
import subprocess

from iterative_less import config, logger_config
from iterative_less.toolkit import (
    run_finetuning,
    calculate_gradients,
    calculate_mmlu_validation_gradients,
    run_evaluation
)

# --- 配置区域 ---
STRATEGIES_TO_RUN = [
    "random",
    "fixed_less",
    "dynamic_less",
    "fixed_less_imputation",
    "dynamic_less_imputation",
]
SHUTDOWN_AFTER_RUN = True
# ---

# --- 辅助函数 ---
def _run_command(command: str):
    logger.info(f"即将执行命令:\n---\n{command.strip()}\n---")
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', cwd=config.PROJECT_ROOT)
        log_lines = []
        for output in process.stdout:
            line = output.strip()
            if line:
                log_lines.append(line)
                logger.debug(f"[Subprocess]: {line}")
        process.wait()
        retcode = process.poll()
        if retcode != 0:
            logger.error(f"命令执行失败，返回码: {retcode}。")
            logger.error("--- 子进程的完整输出日志如下 ---")
            for line in log_lines: logger.error(line)
            logger.error("--- 日志结束 ---")
            raise subprocess.CalledProcessError(retcode, command)
        logger.info("命令执行成功。")
        return log_lines
    except Exception as e:
        logger.error(f"执行命令时发生未知错误: {e}")
        raise

def load_data_as_df(data_file):
    logger.info(f"正在从 {data_file} 加载数据...")
    records = [json.loads(line) for line in open(data_file, 'r', encoding='utf-8')]
    df = pd.DataFrame(records)
    if 'original_index' not in df.columns:
        df['original_index'] = df.index
    logger.info(f"成功加载 {len(df)} 条记录。")
    return df

def save_df_to_jsonl(df, file_path, columns_to_save, rename_map=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_to_save = df[columns_to_save].copy()
    if rename_map:
        df_to_save.rename(columns=rename_map, inplace=True)
    df_to_save.to_json(file_path, orient='records', lines=True, force_ascii=False)
    logger.info(f"成功保存 {len(df_to_save)} 条数据到 {file_path}")

def normalize_scores(scores_series):
    min_val, max_val = scores_series.min(), scores_series.max()
    if max_val == min_val: return pd.Series(0.5, index=scores_series.index)
    return (scores_series - min_val) / (max_val - min_val)

def manage_disk_space(strategy_dir, current_phase, keep_last_n=1):
    """主动清理旧阶段的模型和梯度文件以节省数据盘空间"""
    phase_to_delete = current_phase - keep_last_n
    if phase_to_delete > 0:
        dir_to_delete = os.path.join(strategy_dir, f"phase_{phase_to_delete}")
        if os.path.exists(dir_to_delete):
            logger.info(f"磁盘管理：正在删除旧目录 {dir_to_delete}")
            # 只删除最占空间的部分
            shutil.rmtree(os.path.join(dir_to_delete, "models"), ignore_errors=True)
            shutil.rmtree(os.path.join(dir_to_delete, "temp_less_grads"), ignore_errors=True)
            logger.info(f"已清理 {dir_to_delete} 内的模型和梯度。")

# --- 核心评分与推理函数 ---

def run_mt5_finetuning(train_file_path, output_dir, num_samples):
    logger.info(f"在 {num_samples} 条数据上微调mT5模型...")
    # --- 【核心修改 1】所有路径都基于config ---
    cmd = (f"python train_imputation_model.py "
           f"--train_file {train_file_path} "
           f"--output_dir {output_dir}")
    _run_command(cmd)
    return output_dir

@torch.no_grad()
def run_entropy_scoring(mt5_model_path, df_to_score, batch_size=32): # 增大批次
    logger.info(f"开始使用 {mt5_model_path} 计算 {len(df_to_score)} 条数据的预测熵...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(mt5_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(mt5_model_path)
    model.eval()
    
    results = []
    for i in tqdm(range(0, len(df_to_score), batch_size), desc="Entropy Scoring"):
        batch_df = df_to_score.iloc[i:i+batch_size]
        original_indices = batch_df['original_index'].tolist()
        input_texts = [row['masked_text'].split(" Target: ")[0] for _, row in batch_df.iterrows()]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
        outputs = model.generate(**inputs, max_length=256, return_dict_in_generate=True, output_scores=True)
        
        batch_entropies = [[] for _ in range(len(batch_df))]
        for step_logits in outputs.scores:
            probs = F.softmax(step_logits, dim=-1)
            log_probs = F.log_softmax(step_logits, dim=-1)
            step_entropies = -torch.sum(probs * log_probs, dim=-1)
            for j, entropy in enumerate(step_entropies):
                batch_entropies[j].append(entropy.item())

        for j in range(len(batch_df)):
            avg_entropy = np.mean(batch_entropies[j]) if batch_entropies[j] else 0.0
            results.append({"original_index": original_indices[j], "imputation_score": avg_entropy})
            
    del model, tokenizer
    torch.cuda.empty_cache()
    return pd.DataFrame(results)

@torch.no_grad()
def run_imputation(mt5_model_path, df_to_impute, batch_size=32): # 增大批次
    logger.info(f"开始使用 {mt5_model_path} 对 {len(df_to_impute)} 条数据进行插补...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(mt5_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(mt5_model_path)
    model.eval()
    
    imputed_texts = []
    for i in tqdm(range(0, len(df_to_impute), batch_size), desc="Imputing Text"):
        batch_df = df_to_impute.iloc[i:i+batch_size]
        input_texts = [row['masked_text'].split(" Target: ")[0] for _, row in batch_df.iterrows()]
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, max_length=512, truncation=True).to(device)
        generated_ids = model.generate(**inputs, max_length=512, num_beams=1, early_stopping=True)
        batch_imputed_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        imputed_texts.extend(batch_imputed_texts)
        
    df_imputed = df_to_impute.copy()
    df_imputed['imputed_text'] = imputed_texts
    del model, tokenizer
    torch.cuda.empty_cache()
    return df_imputed

@torch.no_grad()
def get_less_scores(llama_model_path, val_grad_path, text_df_with_imputed_text, output_dir):
    num_samples = len(text_df_with_imputed_text)
    if num_samples == 0:
        return pd.DataFrame(columns=["original_index", "less_score"])
    logger.info(f"开始使用 {os.path.basename(llama_model_path)} 计算 {num_samples} 条数据的LESS分数...")
    
    # --- 【核心修改 2】所有中间文件路径都基于传入的output_dir ---
    temp_less_input_file = os.path.join(output_dir, "temp_imputed_for_less.jsonl")
    save_df_to_jsonl(text_df_with_imputed_text, temp_less_input_file, columns_to_save=['original_index', 'imputed_text'], rename_map={'imputed_text': 'text'})
    
    train_grad_output_path = os.path.join(output_dir, "temp_less_grads")
    train_grad_path = calculate_gradients(
        model_path=llama_model_path,
        data_file=temp_less_input_file,
        num_samples=num_samples,
        output_path=train_grad_output_path,
        grad_type="sgd", # 使用sgd进行快速计算
        is_lora=True,
        base_model_path=config.BASE_MODEL_NAME
    )
    
    # --- 【核心修改 3】修正算法：使用余弦相似度 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_grads = torch.load(train_grad_path, map_location=device)
    val_grads = torch.load(val_grad_path, map_location=device)
    
    train_grads_normalized = F.normalize(train_grads, p=2, dim=1)
    val_grads_normalized = F.normalize(val_grads, p=2, dim=1)
    influence_scores = torch.matmul(train_grads_normalized, val_grads_normalized.T)

    final_scores = influence_scores.max(dim=1)[0].cpu().numpy()
    scores_df = pd.DataFrame({"original_index": text_df_with_imputed_text['original_index'], "less_score": final_scores})
    
    # --- 【核心修改 4】主动资源清理 ---
    del train_grads, val_grads, train_grads_normalized, val_grads_normalized, influence_scores
    torch.cuda.empty_cache()
    shutil.rmtree(train_grad_output_path, ignore_errors=True)
    os.remove(temp_less_input_file)
    return scores_df

# --- 主程序 ---
def main():
    os.makedirs(config.ITERATIVE_OUTPUT_DIR, exist_ok=True)
    global logger
    logger = logger_config.setup_logger(os.path.join(config.ITERATIVE_OUTPUT_DIR, "run.log"))
    
    logger.info(f"\n{'='*80}\n实验开始: {config.EXPERIMENT_NAME}\n{'='*80}")
    logger.info(f"将运行以下策略: {', '.join(STRATEGIES_TO_RUN)}")

    try:
        logger.info("--- 阶段 0: 准备数据与预热模型 ---")
        warmup_df = load_data_as_df(config.WARMUP_POOL_FILENAME)
        candidate_df = load_data_as_df(config.CANDIDATE_POOL_FILENAME)
        
        warmup_llama_train_file = os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_llama_train.jsonl")
        save_df_to_jsonl(warmup_df, warmup_llama_train_file, columns_to_save=['full_text'], rename_map={'full_text': 'text'})
        llama_m_warmup_path = run_finetuning(
            base_model_name=config.BASE_MODEL_NAME, 
            train_file=warmup_llama_train_file,
            output_dir=os.path.join(config.ITERATIVE_OUTPUT_DIR, "models", "llama_m_warmup"),
            num_train_samples=len(warmup_df)
        )

        warmup_mt5_train_file = os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_mt5_train.jsonl")
        save_df_to_jsonl(warmup_df, warmup_mt5_train_file, columns_to_save=['masked_text'], rename_map={'masked_text': 'text'})
        mt5_s_warmup_path = run_mt5_finetuning(
            train_file_path=warmup_mt5_train_file, 
            output_dir=os.path.join(config.ITERATIVE_OUTPUT_DIR, "models", "mt5_s_warmup"),
            num_samples=len(warmup_df)
        )
        
        baseline_dev_acc, baseline_dev_loss = run_evaluation(model_path=llama_m_warmup_path, eval_output_dir=os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_eval"), eval_split="dev")
        
        logger.info("计算全局MMLU验证集梯度...")
        val_grad_path = calculate_mmlu_validation_gradients(
            model_path=llama_m_warmup_path,
            output_path=os.path.join(config.ITERATIVE_OUTPUT_DIR, "val_grads"),
            is_lora=True,
            base_model_path=config.BASE_MODEL_NAME
        )

        logger.info("计算全局固定LESS分数 (在预热模型插补的文本上)...")
        imputed_candidate_df = run_imputation(mt5_s_warmup_path, candidate_df)
        fixed_less_scores_df = get_less_scores(
            llama_m_warmup_path, val_grad_path, imputed_candidate_df, os.path.join(config.ITERATIVE_OUTPUT_DIR, "fixed_scores_calculation")
        ).rename(columns={'less_score': 'fixed_less_score'})
        del imputed_candidate_df
        torch.cuda.empty_cache()

        strategy_states = {s: {"selected_indices": set(), "llama_model_path": llama_m_warmup_path, "mt5_model_path": mt5_s_warmup_path} for s in STRATEGIES_TO_RUN}
        experiment_log = [{"phase": 0, "strategy": "baseline", "dev_acc": baseline_dev_acc, "dev_loss": baseline_dev_loss, "total_samples": len(warmup_df)}]

        for i in range(1, config.NUM_ITERATIONS + 1):
            phase = i
            for strategy in STRATEGIES_TO_RUN:
                state = strategy_states[strategy]
                logger.info(f"\n{'='*80}\n--- 迭代 {phase}, 策略 '{strategy}' ---")
                
                num_to_add_this_round = config.SELECTION_SIZE_PER_ITERATION
                available_indices = list(set(candidate_df['original_index']) - state["selected_indices"])
                available_df = candidate_df[candidate_df['original_index'].isin(available_indices)].copy().reset_index(drop=True)
                
                if len(available_df) == 0:
                    logger.warning("候选池已空，跳过此策略。")
                    continue
                
                strategy_dir = os.path.join(config.ITERATIVE_OUTPUT_DIR, strategy)
                phase_dir = os.path.join(strategy_dir, f"phase_{phase}")
                os.makedirs(phase_dir, exist_ok=True)
                
                scores_df = pd.DataFrame()
                
                if strategy == "random":
                    pass
                else:
                    if strategy == "fixed_less":
                        scores_df = fixed_less_scores_df[fixed_less_scores_df['original_index'].isin(available_indices)]
                        scores_df['final_score'] = scores_df['fixed_less_score']
                    elif strategy == "dynamic_less":
                        imputed_df = run_imputation(state['mt5_model_path'], available_df)
                        scores_df = get_less_scores(state['llama_model_path'], val_grad_path, imputed_df, phase_dir)
                        scores_df['final_score'] = scores_df['less_score']
                    elif strategy == "fixed_less_imputation":
                        imp_scores_df = run_entropy_scoring(state['mt5_model_path'], available_df)
                        scores_df = pd.merge(fixed_less_scores_df, imp_scores_df, on='original_index', how='inner')
                        scores_df['norm_less'] = normalize_scores(scores_df['fixed_less_score'])
                        scores_df['norm_imp'] = normalize_scores(scores_df['imputation_score'])
                        scores_df['final_score'] = 0.5 * scores_df['norm_less'] + 0.5 * scores_df['norm_imp']
                    elif strategy == "dynamic_less_imputation":
                        imputed_df = run_imputation(state['mt5_model_path'], available_df)
                        less_scores_df = get_less_scores(state['llama_model_path'], val_grad_path, imputed_df, phase_dir)
                        imp_scores_df = run_entropy_scoring(state['mt5_model_path'], available_df)
                        scores_df = pd.merge(less_scores_df, imp_scores_df, on='original_index', how='inner')
                        scores_df['norm_less'] = normalize_scores(scores_df['less_score'])
                        scores_df['norm_imp'] = normalize_scores(scores_df['imputation_score'])
                        scores_df['final_score'] = 0.5 * scores_df['norm_less'] + 0.5 * scores_df['norm_imp']

                if strategy == "random":
                    newly_selected_indices = random.sample(available_indices, k=min(num_to_add_this_round, len(available_indices)))
                else:
                    newly_selected_indices = scores_df.nlargest(num_to_add_this_round, 'final_score')['original_index'].tolist()

                state["selected_indices"].update(newly_selected_indices)
                models_dir = os.path.join(phase_dir, "models")
                
                # --- 【核心修改 5】所有策略都混合训练，确保公平对比 ---
                # 随机选择一小部分预热数据作为“正则化项”
                num_warmup_regularizers = min(len(warmup_df), 500)
                warmup_subset_for_reg = warmup_df.sample(n=num_warmup_regularizers, random_state=config.RANDOM_SEED + phase)
                
                # 合并 Llama 训练数据
                current_selected_df = candidate_df[candidate_df['original_index'].isin(state["selected_indices"])]
                llama_train_df = pd.concat([
                    current_selected_df[['full_text']].rename(columns={'full_text': 'text'}),
                    warmup_subset_for_reg[['full_text']].rename(columns={'full_text': 'text'})
                ])
                llama_train_file = os.path.join(phase_dir, "llama_train_data.jsonl")
                llama_train_df.to_json(llama_train_file, orient='records', lines=True, force_ascii=False)
                
                new_llama_path = run_finetuning(
                    base_model_name=config.BASE_MODEL_NAME,
                    train_file=llama_train_file,
                    output_dir=os.path.join(models_dir, "llama"),
                    num_train_samples=len(llama_train_df)
                )
                state["llama_model_path"] = new_llama_path

                if "imputation" in strategy or "less" in strategy:
                    mt5_train_df = pd.concat([
                        current_selected_df[['masked_text']],
                        warmup_subset_for_reg[['masked_text']]
                    ])
                    mt5_train_file = os.path.join(phase_dir, "mt5_train_data.jsonl")
                    mt5_train_df.to_json(mt5_train_file, orient='records', lines=True, force_ascii=False)
                    new_mt5_path = run_mt5_finetuning(
                        train_file_path=mt5_train_file,
                        output_dir=os.path.join(models_dir, "mt5"),
                        num_samples=len(mt5_train_df)
                    )
                    state["mt5_model_path"] = new_mt5_path
                
                dev_acc, dev_loss = run_evaluation(model_path=new_llama_path, eval_output_dir=os.path.join(phase_dir, "eval_dev"), eval_split="dev")
                test_acc, test_loss = (None, None)
                if phase == config.NUM_ITERATIONS:
                    test_acc, test_loss = run_evaluation(model_path=new_llama_path, eval_output_dir=os.path.join(phase_dir, "eval_test"), eval_split="test")

                log_entry = {"phase": phase, "strategy": strategy, "dev_acc": dev_acc, "dev_loss": dev_loss, "test_acc": test_acc, "test_loss": test_loss, "total_samples": len(current_selected_df)}
                experiment_log.append(log_entry)
                
                manage_disk_space(strategy_dir, phase)
        
        final_log_path = os.path.join(config.ITERATIVE_OUTPUT_DIR, "experiment_results.json")
        with open(final_log_path, 'w', encoding='utf-8') as f:
            json.dump(experiment_log, f, indent=2, ensure_ascii=False)
        logger.info(f"所有策略及阶段完成！统一结果摘要已保存至: {final_log_path}")

    except Exception as e:
        logger.error("实验流程发生致命错误!", exc_info=True)
    finally:
        if SHUTDOWN_AFTER_RUN: os.system("shutdown")

if __name__ == "__main__":
    main()