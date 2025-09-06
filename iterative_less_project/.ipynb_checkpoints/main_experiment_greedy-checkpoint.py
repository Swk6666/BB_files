import os
import sys
import logging
# --- [终极死锁修复] ---
import torch.multiprocessing as mp
# --- [修复结束] ---

def main():
    # --- [终极死锁修复] ---
    import json
    import torch
    import pandas as pd
    import random
    import time
    import shutil
    import subprocess
    # --- [修复结束] ---

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from iterative_less import config, logger_config
    from iterative_less.toolkit import (
        run_finetuning, run_evaluation, verify_optimizer_health,
        calculate_gradients, calculate_mmlu_validation_gradients, run_matching_subprocess
    )

    # ... (The rest of the file is identical to the one from my previous reply) ...
    # (The only change is adding the mp.set_start_method('spawn') call at the bottom)
    def _run_greedy_subprocess(command: list, description: str):
        logger.info(f"即将执行 [{description}]...")
        cmd_str = " ".join(command)
        logger.debug(f"完整命令:\n---\n{cmd_str}\n---")
        try:
            process = subprocess.Popen(
                cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', cwd=config.PROJECT_ROOT
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                logger.error(f"命令 [{description}] 执行失败，返回码: {process.returncode}。")
                if stdout:
                    logger.error("--- 子进程 STDOUT ---")
                    for line in stdout.strip().split('\n'): logger.error(line)
                if stderr:
                    logger.error("--- 子进程 STDERR ---")
                    for line in stderr.strip().split('\n'): logger.error(line)
                raise subprocess.CalledProcessError(process.returncode, cmd_str)
            logger.info(f"✅ 命令 [{description}] 执行成功。")
        except Exception as e:
            logger.error(f"执行命令时发生未知错误: {e}")
            raise

    def run_imputation_subprocess(mt5_model_path, input_df, phase_dir):
        input_path = os.path.join(phase_dir, f"imputation_input_{int(time.time())}.jsonl")
        output_path = os.path.join(phase_dir, f"imputation_output_{int(time.time())}.jsonl")
        input_df.to_json(input_path, orient='records', lines=True, force_ascii=False)
        
        command = [
            "python", "-m", "run_imputation",
            "--mt5_model_path", mt5_model_path,
            "--input_df_path", input_path,
            "--output_df_path", output_path
        ]
        _run_greedy_subprocess(command, "文本插补 (Imputation)")
        
        if not os.path.exists(output_path):
            raise FileNotFoundError("Imputation 脚本执行后，未找到预期的输出文件！")
        return pd.read_json(output_path, lines=True)

    def run_entropy_scoring_subprocess(mt5_model_path, input_df, phase_dir):
        input_path = os.path.join(phase_dir, f"entropy_input_{int(time.time())}.jsonl")
        output_path = os.path.join(phase_dir, f"entropy_output_{int(time.time())}.jsonl")
        input_df.to_json(input_path, orient='records', lines=True, force_ascii=False)
        
        command = [
            "python", "-m", "run_entropy_scoring",
            "--mt5_model_path", mt5_model_path,
            "--input_df_path", input_path,
            "--output_scores_path", output_path
        ]
        _run_greedy_subprocess(command, "熵评分 (Entropy Scoring)")
        
        if not os.path.exists(output_path):
            raise FileNotFoundError("Entropy scoring 脚本执行后，未找到预期的输出文件！")
        return pd.read_json(output_path, lines=True)

    def run_less_scoring_subprocess(llama_model_path, val_grad_path, input_df, phase_dir):
        scoring_temp_dir = os.path.join(phase_dir, "scoring_temp_files")
        os.makedirs(scoring_temp_dir, exist_ok=True)
        
        temp_train_file = os.path.join(scoring_temp_dir, "temp_grad_input.jsonl")
        # 确保 original_index 被传递
        input_df[['text', 'original_index']].to_json(temp_train_file, orient='records', lines=True, force_ascii=False)
        
        train_grad_path = calculate_gradients(
            model_path=llama_model_path, data_file=temp_train_file, num_samples=len(input_df),
            output_path=os.path.join(scoring_temp_dir, "temp_grads"),
            grad_type=config.GRADIENT_TYPE_TRAIN_POOL_ADAM, is_lora=True, base_model_path=config.BASE_MODEL_NAME
        )
        
        _, scores_df = run_matching_subprocess(
            train_grad_path=train_grad_path, val_grad_path=val_grad_path,
            selection_size=len(input_df), num_train_samples=len(input_df),
            phase_dir=scoring_temp_dir
        )
        
        # 将原始索引合并回分数DataFrame
        scores_df['original_index'] = input_df['original_index'].values
        return scores_df.rename(columns={'score': 'less_score'})

    STRATEGIES_TO_RUN = ["random", "fixed_less", "dynamic_less", "fixed_less_imputation", "dynamic_less_imputation"]
    SHUTDOWN_AFTER_RUN = True
    WARMUP_MIX_RATIO = 1.0

    def load_data_as_df(data_file):
        logger.info(f"正在从 {data_file} 加载数据...")
        records = [json.loads(line) for line in open(data_file, 'r', encoding='utf-8')]
        df = pd.DataFrame(records)
        if 'original_index' not in df.columns: df['original_index'] = df.index
        logger.info(f"✅ 成功加载 {len(df)} 条记录。"); return df

    def normalize_scores(scores_series):
        min_val, max_val = scores_series.min(), scores_series.max()
        if pd.isna(min_val) or pd.isna(max_val):
            logger.warning("分数序列中包含NaN，无法归一化。将返回0.5。")
            return pd.Series(0.5, index=scores_series.index)
        if max_val == min_val: return pd.Series(0.5, index=scores_series.index)
        return (scores_series - min_val) / (max_val - min_val)

    def manage_disk_space(strategy_dir, current_phase, keep_last_n=1):
        phase_to_delete = current_phase - keep_last_n
        if phase_to_delete > 0:
            dir_to_delete = os.path.join(strategy_dir, f"phase_{phase_to_delete}")
            if os.path.exists(dir_to_delete):
                logger.info(f"磁盘管理：正在删除旧目录 {dir_to_delete}")
                shutil.rmtree(os.path.join(dir_to_delete, "models"), ignore_errors=True)
                shutil.rmtree(os.path.join(dir_to_delete, "scoring_temp_files"), ignore_errors=True)
                logger.info(f"✅ 已清理 {dir_to_delete} 内的模型和临时文件。")

    def log_score_stats(scores_df: pd.DataFrame, score_column: str, strategy_name: str, phase: int):
        if scores_df is None or scores_df.empty or score_column not in scores_df.columns:
            logger.warning(f"[{strategy_name} Phase {phase}] 无法计算分数统计，数据为空。")
            return
        stats = scores_df[score_column].describe()
        mean_score, min_score, max_score, std_dev = stats.get('mean', float('nan')), stats.get('min', float('nan')), stats.get('max', float('nan')), stats.get('std', float('nan'))
        logger.info(f"--- 分数监控 [{strategy_name} Phase {phase}] ---")
        logger.info(f"    - 平均分 (Mean): {mean_score:.6f}, 标准差 (Std):  {std_dev:.6f}")
        logger.info(f"    - 最小/大分: {min_score:.6f} / {max_score:.6f}, 总点数: {len(scores_df)}")
        if pd.isna(mean_score) or (mean_score == 0 and std_dev == 0):
            logger.critical(f"严重警告：[{strategy_name} Phase {phase}] 的分数全为零或NaN！")

    os.makedirs(config.ITERATIVE_OUTPUT_DIR, exist_ok=True)
    global logger; logger = logger_config.setup_logger(os.path.join(config.ITERATIVE_OUTPUT_DIR, "run.log"))
    
    try:
        logger.info(f"\n{'='*80}\n实验开始: {config.EXPERIMENT_NAME}\n{'='*80}")
        logger.info(f"将运行以下策略: {', '.join(STRATEGIES_TO_RUN)}")
        logger.info("--- 阶段 0: 准备数据与预热模型 ---")
        warmup_df = load_data_as_df(config.WARMUP_POOL_FILENAME)
        candidate_df = load_data_as_df(config.CANDIDATE_POOL_FILENAME)
        warmup_llama_train_file = os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_llama_train.jsonl")
        warmup_df[['text']].to_json(warmup_llama_train_file, orient='records', lines=True, force_ascii=False)
        llama_m_warmup_path = run_finetuning(base_model_name=config.BASE_MODEL_NAME, train_file=warmup_llama_train_file, output_dir=os.path.join(config.ITERATIVE_OUTPUT_DIR, "models", "llama_m_warmup"), num_train_samples=len(warmup_df))
        verify_optimizer_health(model_path=llama_m_warmup_path, base_model_path=config.BASE_MODEL_NAME)
        mt5_s_warmup_path = "path_to_your_trained_mt5_model" 
        baseline_dev_acc, baseline_dev_loss = run_evaluation(model_path=llama_m_warmup_path, eval_output_dir=os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_eval"), eval_split="dev")
        logger.info("计算全局MMLU验证集梯度...")
        val_grad_path = calculate_mmlu_validation_gradients(model_path=llama_m_warmup_path, output_path=os.path.join(config.ITERATIVE_OUTPUT_DIR, "val_grads"), is_lora=True, base_model_path=config.BASE_MODEL_NAME)
        logger.info("计算全局固定LESS分数...")
        fixed_less_scores_df = run_less_scoring_subprocess(llama_m_warmup_path, val_grad_path, candidate_df, os.path.join(config.ITERATIVE_OUTPUT_DIR, "fixed_scores_calculation")).rename(columns={'less_score': 'fixed_less_score'})
        log_score_stats(fixed_less_scores_df, 'fixed_less_score', 'Fixed Scorer (Global)', 0)
        strategy_states = {s: {"selected_indices": set(), "llama_model_path": llama_m_warmup_path, "mt5_model_path": mt5_s_warmup_path} for s in STRATEGIES_TO_RUN}
        experiment_log = [{"phase": 0, "strategy": "baseline", "dev_acc": baseline_dev_acc, "dev_loss": baseline_dev_loss, "total_samples": len(warmup_df)}]
        for i in range(1, config.NUM_ITERATIONS + 1):
            phase = i; num_to_select_total = phase * config.SELECTION_SIZE_PER_ITERATION
            logger.info(f"\n{'='*80}\n--- 迭代 {phase}, (目标选择量: {num_to_select_total}) ---")
            warmup_subset_for_phase = warmup_df.sample(n=int(len(warmup_df) * WARMUP_MIX_RATIO), random_state=config.RANDOM_SEED + phase)
            for strategy in STRATEGIES_TO_RUN:
                state = strategy_states[strategy]
                logger.info(f"\n--- 正在执行策略 '{strategy}' ---")
                num_to_add_this_round = num_to_select_total - len(state["selected_indices"])
                if num_to_add_this_round <= 0: logger.info("已达到目标样本数，跳过选择。"); continue
                available_indices = list(set(candidate_df['original_index']) - state["selected_indices"])
                if not available_indices: logger.warning("候选池已空。"); continue
                available_df = candidate_df[candidate_df['original_index'].isin(available_indices)].copy().reset_index(drop=True)
                strategy_dir = os.path.join(config.ITERATIVE_OUTPUT_DIR, strategy); phase_dir = os.path.join(strategy_dir, f"phase_{phase}"); os.makedirs(phase_dir, exist_ok=True)
                scores_df = pd.DataFrame()
                if strategy == "random": pass
                elif strategy == "fixed_less":
                    scores_df = fixed_less_scores_df[fixed_less_scores_df['original_index'].isin(available_indices)].copy()
                    scores_df['final_score'] = scores_df['fixed_less_score']
                elif strategy == "dynamic_less":
                    scores_df = run_less_scoring_subprocess(state['llama_model_path'], val_grad_path, available_df, phase_dir)
                    scores_df['final_score'] = scores_df['less_score']
                elif "imputation" in strategy:
                    imp_scores_df = run_entropy_scoring_subprocess(state['mt5_model_path'], available_df, phase_dir)
                    if strategy == "fixed_less_imputation":
                        scores_df = pd.merge(fixed_less_scores_df, imp_scores_df, on='original_index', how='inner')
                        scores_df['norm_less'] = normalize_scores(scores_df['fixed_less_score'])
                    else:
                        less_scores_df = run_less_scoring_subprocess(state['llama_model_path'], val_grad_path, available_df, phase_dir)
                        scores_df = pd.merge(less_scores_df, imp_scores_df, on='original_index', how='inner')
                        scores_df['norm_less'] = normalize_scores(scores_df['less_score'])
                    scores_df['norm_imp'] = normalize_scores(scores_df['imputation_score'])
                    scores_df['final_score'] = 0.5 * scores_df['norm_less'] + 0.5 * scores_df['norm_imp']
                if not scores_df.empty:
                    log_score_stats(scores_df, 'final_score', strategy, phase)
                if strategy == "random": newly_selected_indices = random.sample(available_indices, k=min(num_to_add_this_round, len(available_indices)))
                else: newly_selected_indices = scores_df.nlargest(num_to_add_this_round, 'final_score')['original_index'].tolist()
                state["selected_indices"].update(newly_selected_indices)
                models_dir = os.path.join(phase_dir, "models")
                current_selected_df = candidate_df[candidate_df['original_index'].isin(state["selected_indices"])]
                llama_train_df = pd.concat([current_selected_df[['text']].copy(), warmup_subset_for_phase[['text']].copy()])
                llama_train_file = os.path.join(phase_dir, "llama_train_data.jsonl"); llama_train_df.to_json(llama_train_file, orient='records', lines=True, force_ascii=False)
                new_llama_path = run_finetuning(base_model_name=config.BASE_MODEL_NAME, train_file=llama_train_file, output_dir=os.path.join(models_dir, "llama"), num_train_samples=len(llama_train_df))
                verify_optimizer_health(model_path=new_llama_path, base_model_path=config.BASE_MODEL_NAME)
                state["llama_model_path"] = new_llama_path
                dev_acc, dev_loss = run_evaluation(model_path=new_llama_path, eval_output_dir=os.path.join(phase_dir, "eval_dev"), eval_split="dev")
                test_acc, test_loss = (None, None)
                if phase == config.NUM_ITERATIONS: test_acc, test_loss = run_evaluation(model_path=new_llama_path, eval_output_dir=os.path.join(phase_dir, "eval_test"), eval_split="test")
                log_entry = {"phase": phase, "strategy": strategy, "dev_acc": dev_acc, "dev_loss": dev_loss, "test_acc": test_acc, "test_loss": test_loss, "total_samples": len(current_selected_df)}
                experiment_log.append(log_entry)
                manage_disk_space(strategy_dir, phase)
        final_log_path = os.path.join(config.ITERATIVE_OUTPUT_DIR, "experiment_results.json")
        with open(final_log_path, 'w', encoding='utf-8') as f: json.dump(experiment_log, f, indent=2, ensure_ascii=False)
        logger.info(f"🎉 所有策略及阶段完成！统一结果摘要已保存至: {final_log_path}")
    except Exception as e:
        logger.critical("实验流程发生致命错误!", exc_info=True)
    finally:
        if SHUTDOWN_AFTER_RUN:
            logger.warning("实验结束，将在60秒后自动关机...")
            os.system("shutdown")

if __name__ == "__main__":
    # --- [终极死锁修复] ---
    mp.set_start_method('spawn', force=True)
    # --- [修复结束] ---
    main()