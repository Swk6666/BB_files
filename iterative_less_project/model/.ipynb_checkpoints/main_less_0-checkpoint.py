# /root/iterative_less_project/main_less.py
import os
import json
import logging
import torch
import pandas as pd
import random
import time
import shutil
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from iterative_less import config, logger_config
from iterative_less.data_manager import DataManager
from iterative_less.toolkit import (
    run_finetuning, calculate_gradients, calculate_mmlu_validation_gradients,
    run_evaluation, verify_optimizer_health, run_matching_subprocess # 引入新的子流程函数
)

SHUTDOWN_AFTER_RUN = True
WARMUP_MIX_RATIO = 0.55

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

def main():
    os.makedirs(config.ITERATIVE_OUTPUT_DIR, exist_ok=True)
    global logger; logger = logger_config.setup_logger(os.path.join(config.ITERATIVE_OUTPUT_DIR, "run.log"))

    try:
        logger.info(f"\n{'='*50}\n实验开始: {config.EXPERIMENT_NAME}\n{'='*50}")
        data_manager = DataManager(config.CANDIDATE_POOL_FILENAME)
        warmup_df = pd.read_json(config.WARMUP_POOL_FILENAME, lines=True)
        results_log = []

        logger.info("\n--- 阶段 1: 初始模型预热 ---")
        warmup_model_path = os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_model")
        if not os.path.exists(os.path.join(warmup_model_path, "adapter_config.json")):
            run_finetuning(base_model_name=config.BASE_MODEL_NAME, train_file=config.WARMUP_POOL_FILENAME, output_dir=warmup_model_path, num_train_samples=len(warmup_df))
        verify_optimizer_health(model_path=warmup_model_path, base_model_path=config.BASE_MODEL_NAME)
        
        warmup_dev_acc, warmup_dev_loss = run_evaluation(model_path=warmup_model_path, eval_output_dir=os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_eval"), eval_split="dev")
        results_log.append({"strategy": "warmup_baseline", "phase": 0, "accuracy_dev": warmup_dev_acc, "loss_dev": warmup_dev_loss, "total_selected_samples": 0})
        logger.info(f"✅ 预热模型评估完成。Dev Acc: {warmup_dev_acc:.4f}, Dev Loss: {warmup_dev_loss:.4f}")

        logger.info("\n--- 阶段 2: 一次性计算 'Fixed Scorer' 全局梯度与排名 ---")
        one_shot_dir = os.path.join(config.ITERATIVE_OUTPUT_DIR, "one_shot_files")
        os.makedirs(one_shot_dir, exist_ok=True)
        train_grad_file = calculate_gradients(model_path=warmup_model_path, data_file=config.CANDIDATE_POOL_FILENAME, num_samples=data_manager.num_total_samples, output_path=os.path.join(one_shot_dir, "grads_train_pool"), grad_type=config.GRADIENT_TYPE_TRAIN_POOL_ADAM, is_lora=True, base_model_path=config.BASE_MODEL_NAME)
        val_grad_file = calculate_mmlu_validation_gradients(model_path=warmup_model_path, output_path=os.path.join(one_shot_dir, "grads_validation"), is_lora=True, base_model_path=config.BASE_MODEL_NAME)
        
        less_global_ranked_indices, fixed_scores_df = run_matching_subprocess(
            train_grad_path=train_grad_file, val_grad_path=val_grad_file, 
            selection_size=data_manager.num_total_samples, num_train_samples=data_manager.num_total_samples,
            phase_dir=one_shot_dir
        )
        
        log_score_stats(fixed_scores_df, 'score', 'Fixed Scorer (Global)', 0)
        fixed_scores_df.to_json(os.path.join(config.ITERATIVE_OUTPUT_DIR, "fixed_scorer_global_scores.jsonl"), orient="records", lines=True)

        strategy_states = {"fixed_scorer": {}, "dynamic_scorer": {"model_path": warmup_model_path, "selected_indices": set()}, "random": {}}

        for i in range(config.NUM_ITERATIONS):
            phase = i + 1; num_train_samples = (i + 1) * config.SELECTION_SIZE_PER_ITERATION
            logger.info(f"\n{'='*25} 训练阶段 {phase}/{config.NUM_ITERATIONS} (目标: {num_train_samples}样本) {'='*25}")
            
            warmup_subset_for_phase = warmup_df.sample(n=int(len(warmup_df) * WARMUP_MIX_RATIO), random_state=config.RANDOM_SEED + phase)

            for strategy in ["fixed_scorer", "dynamic_scorer", "random"]:
                logger.info(f"\n--- 正在执行策略: [{strategy}] ---")
                phase_dir = os.path.join(config.ITERATIVE_OUTPUT_DIR, f"{strategy}/phase_{phase}"); os.makedirs(phase_dir, exist_ok=True)
                
                indices_to_train = []
                if strategy == "fixed_scorer":
                    indices_to_train = less_global_ranked_indices[:num_train_samples]
                elif strategy == "random":
                    all_indices = list(range(data_manager.num_total_samples)); random.seed(config.RANDOM_SEED + phase)
                    indices_to_train = random.sample(all_indices, k=min(num_train_samples, len(all_indices)))
                elif strategy == "dynamic_scorer":
                    scorer_model_path = strategy_states[strategy]["model_path"]; prev_indices = strategy_states[strategy]["selected_indices"]
                    num_to_add = num_train_samples - len(prev_indices)
                    if num_to_add <= 0: logger.info("已达到目标样本数，跳过选择。"); continue
                    
                    remaining_indices = [idx for idx in range(data_manager.num_total_samples) if idx not in prev_indices]
                    if not remaining_indices: logger.warning("候选池已空。"); continue
                    
                    remaining_data = data_manager.all_data.select(remaining_indices)
                    remaining_data_path = os.path.join(phase_dir, "remaining_data.jsonl"); data_manager.save_dataset_to_jsonl(remaining_data, remaining_data_path)
                    
                    train_grad_dyn = calculate_gradients(model_path=scorer_model_path, data_file=remaining_data_path, num_samples=len(remaining_data), output_path=os.path.join(phase_dir, "grads_train_pool"), grad_type=config.GRADIENT_TYPE_TRAIN_POOL_ADAM, is_lora=True, base_model_path=config.BASE_MODEL_NAME)
                    val_grad_dyn = calculate_mmlu_validation_gradients(model_path=scorer_model_path, output_path=os.path.join(phase_dir, "grads_validation"), is_lora=True, base_model_path=config.BASE_MODEL_NAME)
                    
                    new_relative_indices, dyn_scores_df = run_matching_subprocess(
                        train_grad_path=train_grad_dyn, val_grad_path=val_grad_dyn,
                        selection_size=num_to_add, num_train_samples=len(remaining_data),
                        phase_dir=phase_dir
                    )
                    
                    dyn_scores_df['original_index'] = remaining_indices
                    log_score_stats(dyn_scores_df, 'score', 'Dynamic Scorer', phase)
                    dyn_scores_df.to_json(os.path.join(phase_dir, "calculated_scores.jsonl"), orient="records", lines=True)

                    new_global_indices = [remaining_indices[i] for i in new_relative_indices]; indices_to_train = list(prev_indices) + new_global_indices
                
                selected_data_df = data_manager.all_data.select(indices_to_train).to_pandas()
                final_training_df = pd.concat([selected_data_df[['text']].copy(), warmup_subset_for_phase[['text']].copy()])
                train_file_path = os.path.join(phase_dir, "training_data_mixed.jsonl")
                final_training_df.to_json(train_file_path, orient='records', lines=True, force_ascii=False)
                logger.info(f"混合训练集: {len(selected_data_df)} (选择) + {len(warmup_subset_for_phase)} (预热) = {len(final_training_df)} (总)。")

                new_model_path = run_finetuning(base_model_name=config.BASE_MODEL_NAME, train_file=train_file_path, output_dir=os.path.join(phase_dir, "evaluation_model"), num_train_samples=len(final_training_df))
                verify_optimizer_health(model_path=new_model_path, base_model_path=config.BASE_MODEL_NAME)

                dev_acc, dev_loss = run_evaluation(model_path=new_model_path, eval_output_dir=os.path.join(phase_dir, "evaluation_results"), eval_split="dev")
                log_entry = { "strategy": strategy, "phase": phase, "accuracy_dev": dev_acc, "loss_dev": dev_loss, "total_selected_samples": len(indices_to_train) }
                results_log.append(log_entry)
                logger.info(f"✅ 策略 [{strategy}] 阶段 {phase} 完成。Dev Acc: {dev_acc:.4f}, Dev Loss: {dev_loss:.4f}")

                if strategy == "dynamic_scorer":
                    strategy_states[strategy]["model_path"] = new_model_path
                    strategy_states[strategy]["selected_indices"] = set(indices_to_train)
        
        logger.info("\n--- 阶段 4: 对所有策略的最终模型进行 Test 集评估 ---")
        final_test_results = {}
        for strategy in ["fixed_scorer", "dynamic_scorer", "random"]:
            final_phase_dir = os.path.join(config.ITERATIVE_OUTPUT_DIR, f"{strategy}/phase_{config.NUM_ITERATIONS}")
            final_model_path = os.path.join(final_phase_dir, "evaluation_model")
            if os.path.exists(final_model_path):
                eval_dir = os.path.join(final_phase_dir, "final_test_evaluation")
                test_acc, test_loss = run_evaluation(model_path=final_model_path, eval_output_dir=eval_dir, eval_split="test")
                logger.info(f"策略 [{strategy}] 最终 Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}"); final_test_results[strategy] = {"acc": test_acc, "loss": test_loss}
        
        for log_entry in results_log:
            if log_entry.get("phase") == config.NUM_ITERATIONS and log_entry["strategy"] in final_test_results:
                log_entry["accuracy_test"] = final_test_results[log_entry["strategy"]]["acc"]; log_entry["loss_test"] = final_test_results[log_entry["strategy"]]["loss"]

        final_log_path = os.path.join(config.ITERATIVE_OUTPUT_DIR, "final_results_summary.json")
        with open(final_log_path, 'w', encoding='utf-8') as f: json.dump(results_log, f, indent=4, ensure_ascii=False)
        logger.info(f"\n{'='*50}\n🎉 所有策略及阶段完成！统一结果摘要已保存至: {final_log_path}\n{'='*50}")

    except Exception as e:
        logger.critical("实验流程发生致命错误!", exc_info=True)
    finally:
        if SHUTDOWN_AFTER_RUN: os.system("shutdown")

if __name__ == "__main__":
    main()