# /root/iterative_less_project/main_less.py
import os
import json
import logging
import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM
import random
import time
import shutil

from iterative_less import config, logger_config
from iterative_less.data_manager import DataManager
from iterative_less.toolkit import (
    run_finetuning,
    calculate_gradients,
    calculate_mmlu_validation_gradients,
    run_matching_and_select,
    run_evaluation
)

SHUTDOWN_AFTER_RUN = True
WARMUP_MIX_RATIO = 0.6

# --- 函数定义 (保持不变) ---
def _verify_model_loading(base_model_path, lora_adapter_path=None):
    logger.info(f"正在验证模型加载...")
    logger.info(f"  - 基础模型路径: {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    logger.info(f"  - 成功加载基础模型。")
    if lora_adapter_path:
        logger.info(f"  - LoRA 适配器路径: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path, is_trainable=True)
        model.train()
        logger.info("  - 成功加载LoRA适配器。")
    del model
    torch.cuda.empty_cache()
    logger.info("模型加载验证通过，资源已释放。")

def initiate_shutdown(delay=60):
    if SHUTDOWN_AFTER_RUN:
        logger.warning(f"\n{'='*50}\n实验流程结束，将在 {delay} 秒后自动关机...\n{'='*50}")
        try:
            for i in range(delay, 0, -1):
                print(f"\r关机倒计时: {i}秒...", end="", flush=True)
                time.sleep(1)
            print("\n执行关机命令。")
            os.system("shutdown")
        except KeyboardInterrupt:
            logger.info("用户中断！自动关机已取消。")
    else:
        logger.info(f"\n{'='*50}\n实验流程结束，自动关机功能未启用。\n{'='*50}")

def manage_disk_space_for_phase(phase_dir_to_clean):
    if not phase_dir_to_clean or not os.path.exists(phase_dir_to_clean): return
    logger.info(f"开始清理磁盘空间 (中间产物): {phase_dir_to_clean}")
    dynamic_grads_path = os.path.join(phase_dir_to_clean, "dynamic_gradients")
    if os.path.exists(dynamic_grads_path):
        shutil.rmtree(dynamic_grads_path)
        logger.info(f"  - 已删除目录: {dynamic_grads_path}")

def main():
    os.makedirs(config.ITERATIVE_OUTPUT_DIR, exist_ok=True)
    global logger
    logger = logger_config.setup_logger(os.path.join(config.ITERATIVE_OUTPUT_DIR, "run.log"))

    try:
        logger.info(f"\n{'='*50}\n实验开始: {config.EXPERIMENT_NAME}\n{'='*50}")
        logger.info(f"基础模型: {config.BASE_MODEL_NAME}")
        logger.info(f"输出目录: {config.ITERATIVE_OUTPUT_DIR}")
        logger.info(f"混合训练比例 (WARMUP_MIX_RATIO): {WARMUP_MIX_RATIO}")

        data_manager = DataManager(config.CANDIDATE_POOL_FILENAME)
        logger.info(f"加载预热数据池 '{config.WARMUP_POOL_FILENAME}' 用于混合训练...")
        warmup_df = pd.read_json(config.WARMUP_POOL_FILENAME, lines=True)
        results_log = []

        logger.info("\n--- 阶段 1: 初始模型预热 ---")
        warmup_model_path = os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_model")
        if not os.path.exists(os.path.join(warmup_model_path, "adapter_config.json")):
            run_finetuning(base_model_name=config.BASE_MODEL_NAME, train_file=config.WARMUP_POOL_FILENAME, output_dir=warmup_model_path, num_train_samples=len(warmup_df))
        _verify_model_loading(config.BASE_MODEL_NAME, warmup_model_path)
        
        warmup_dev_acc, warmup_dev_loss = run_evaluation(model_path=warmup_model_path, eval_output_dir=os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_eval"), eval_split="dev")
        results_log.append({"strategy": "warmup_baseline", "phase": 0, "accuracy_dev": warmup_dev_acc, "loss_dev": warmup_dev_loss, "total_selected_samples": 0})
        logger.info(f"预热模型评估完成 (基线)。Dev Acc: {warmup_dev_acc:.4f}, Dev Loss: {warmup_dev_loss:.4f}")

        logger.info("\n--- 阶段 2: 一次性计算 'Fixed Scorer' 所需的全局梯度与排名 ---")
        grads_dir = os.path.join(config.ITERATIVE_OUTPUT_DIR, "one_shot_gradients")
        train_grad_file = calculate_gradients(model_path=warmup_model_path, data_file=config.CANDIDATE_POOL_FILENAME, num_samples=data_manager.num_total_samples, output_path=os.path.join(grads_dir, "grads_train_pool"), grad_type=config.GRADIENT_TYPE_TRAIN_POOL_ADAM, is_lora=True, base_model_path=config.BASE_MODEL_NAME)
        
        # --- 【核心修复】 ---
        # 将参数名从 output_dir 改为 output_path，以匹配 toolkit.py 中的函数定义
        val_grad_file = calculate_mmlu_validation_gradients(
            model_path=warmup_model_path,
            output_path=os.path.join(grads_dir, "grads_validation"), # <- 这里已修复
            is_lora=True,
            base_model_path=config.BASE_MODEL_NAME
        )
        # --- 【修复结束】 ---
        
        less_global_ranked_indices = run_matching_and_select(train_grad_path=train_grad_file, val_grad_path=val_grad_file, selection_size=data_manager.num_total_samples, num_train_samples=data_manager.num_total_samples)
        shutil.rmtree(grads_dir, ignore_errors=True)
        torch.cuda.empty_cache()
        
        logger.info("\n--- 阶段 3: 开始分策略进行分阶段训练与评估 ---")
        
        strategy_states = {"fixed_scorer": {"model_path": warmup_model_path, "selected_indices": set()}, "dynamic_scorer": {"model_path": warmup_model_path, "selected_indices": set()}, "random": {"model_path": warmup_model_path, "selected_indices": set()}}

        num_phases = config.NUM_ITERATIONS
        for i in range(num_phases):
            phase = i + 1
            num_train_samples = (i + 1) * config.SELECTION_SIZE_PER_ITERATION
            logger.info(f"\n{'='*25} 训练阶段 {phase}/{num_phases} 开始 (目标选择量: {num_train_samples}) {'='*25}")
            
            num_warmup_to_add = int(len(warmup_df) * WARMUP_MIX_RATIO)
            logger.info(f"本阶段将为所有策略混入 {num_warmup_to_add} 条相同的预热数据用于稳定训练。")
            warmup_subset_for_phase = warmup_df.sample(n=num_warmup_to_add, random_state=config.RANDOM_SEED + phase)

            for strategy in ["fixed_scorer", "dynamic_scorer", "random"]:
                logger.info(f"\n--- 正在执行策略: [{strategy}] ---")
                phase_dir = os.path.join(config.ITERATIVE_OUTPUT_DIR, f"{strategy}/phase_{phase}")
                os.makedirs(phase_dir, exist_ok=True)

                indices_to_train = []
                if strategy == "fixed_scorer":
                    indices_to_train = less_global_ranked_indices[:num_train_samples]
                elif strategy == "random":
                    all_indices = list(range(data_manager.num_total_samples))
                    random.seed(config.RANDOM_SEED + phase)
                    indices_to_train = random.sample(all_indices, k=min(num_train_samples, len(all_indices)))
                elif strategy == "dynamic_scorer":
                    scorer_model = strategy_states[strategy]["model_path"]
                    prev_indices = strategy_states[strategy]["selected_indices"]
                    num_to_add = num_train_samples - len(prev_indices)
                    if num_to_add <= 0: continue
                    remaining_indices = [idx for idx in range(data_manager.num_total_samples) if idx not in prev_indices]
                    remaining_data = data_manager.all_data.select(remaining_indices)
                    if len(remaining_data) == 0: continue
                    remaining_data_path = os.path.join(phase_dir, "remaining_data.jsonl")
                    data_manager.save_dataset_to_jsonl(remaining_data, remaining_data_path)
                    dyn_grads_dir = os.path.join(phase_dir, "dynamic_gradients")
                    train_grad_dyn = calculate_gradients(model_path=scorer_model, data_file=remaining_data_path, num_samples=len(remaining_data), output_path=os.path.join(dyn_grads_dir, "grads_train_pool"), grad_type=config.GRADIENT_TYPE_TRAIN_POOL_ADAM, is_lora=True, base_model_path=config.BASE_MODEL_NAME)
                    val_grad_dyn = calculate_mmlu_validation_gradients(model_path=scorer_model, output_path=os.path.join(dyn_grads_dir, "grads_validation"), is_lora=True, base_model_path=config.BASE_MODEL_NAME)
                    new_relative_indices = run_matching_and_select(train_grad_path=train_grad_dyn, val_grad_path=val_grad_dyn, selection_size=num_to_add, num_train_samples=len(remaining_data))
                    new_global_indices = [remaining_indices[i] for i in new_relative_indices]
                    indices_to_train = list(prev_indices) + new_global_indices
                
                selected_data_df = data_manager.all_data.select(indices_to_train).to_pandas()
                final_training_df = pd.concat([selected_data_df[['text']], warmup_subset_for_phase[['text']]])
                train_file_path = os.path.join(phase_dir, "training_data_mixed.jsonl")
                final_training_df.to_json(train_file_path, orient='records', lines=True, force_ascii=False)
                logger.info(f"混合训练集构建完成：{len(selected_data_df)}条选择数据 + {len(warmup_subset_for_phase)}条预热数据 = {len(final_training_df)}条总训练数据。")

                new_model_path = run_finetuning(base_model_name=config.BASE_MODEL_NAME, train_file=train_file_path, output_dir=os.path.join(phase_dir, "evaluation_model"), num_train_samples=len(final_training_df))
                
                eval_results_dir = os.path.join(phase_dir, "evaluation_results")
                dev_acc, dev_loss = run_evaluation(model_path=new_model_path, eval_output_dir=eval_results_dir, eval_split="dev")
                log_entry = { "strategy": strategy, "phase": phase, "accuracy_dev": dev_acc, "loss_dev": dev_loss, "total_selected_samples": len(indices_to_train) }
                results_log.append(log_entry)
                logger.info(f"策略 [{strategy}] 阶段 {phase} 完成。Dev Acc: {dev_acc:.4f}, Dev Loss: {dev_loss:.4f}")

                prev_model_path_to_clean = strategy_states[strategy]["model_path"]
                strategy_states[strategy]["model_path"] = new_model_path
                strategy_states[strategy]["selected_indices"] = set(indices_to_train)
                
                manage_disk_space_for_phase(phase_dir)
                if prev_model_path_to_clean and prev_model_path_to_clean != warmup_model_path:
                    shutil.rmtree(prev_model_path_to_clean, ignore_errors=True)
                torch.cuda.empty_cache()
        
        logger.info("\n--- 阶段 4: 对所有策略的最终模型进行 Test 集评估 ---")
        final_test_results = {}
        for strategy, state in strategy_states.items():
            final_model_path = state.get("model_path")
            if final_model_path and final_model_path != warmup_model_path:
                eval_dir = os.path.join(config.ITERATIVE_OUTPUT_DIR, f"{strategy}/final_test_evaluation")
                test_acc, test_loss = run_evaluation(model_path=final_model_path, eval_output_dir=eval_dir, eval_split="test")
                logger.info(f"策略 [{strategy}] 最终 Test Acc: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
                final_test_results[strategy] = {"acc": test_acc, "loss": test_loss}
        
        for log_entry in results_log:
            if log_entry.get("phase") == num_phases and log_entry["strategy"] in final_test_results:
                log_entry["accuracy_test"] = final_test_results[log_entry["strategy"]]["acc"]
                log_entry["loss_test"] = final_test_results[log_entry["strategy"]]["loss"]

        final_log_path = os.path.join(config.ITERATIVE_OUTPUT_DIR, "final_results_summary.json")
        with open(final_log_path, 'w', encoding='utf-8') as f:
            json.dump(results_log, f, indent=4, ensure_ascii=False)
        logger.info(f"\n{'='*50}\n所有策略及阶段完成！统一结果摘要已保存至: {final_log_path}\n{'='*50}")

    except Exception as e:
        logger.error("实验流程发生致命错误!", exc_info=True)
    finally:
        initiate_shutdown()

if __name__ == "__main__":
    main()