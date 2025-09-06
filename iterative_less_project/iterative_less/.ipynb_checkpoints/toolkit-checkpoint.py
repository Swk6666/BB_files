# /root/iterative_less_project/iterative_less/toolkit.py
import os
import subprocess
import torch
import logging
import shutil
import json
import pandas as pd
import time
import torch.nn.functional as F
from . import config
from transformers import AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)

# --- 【核心修复】 ---
# 重写命令执行函数，采用 process.communicate() 来彻底避免I/O死锁。
def _run_command_final(command: str, description: str, is_gpu_intensive: bool = True, use_progress_log: bool = False):
    """
    最终修复版的命令执行器。
    它通过等待子进程完成后再用communicate()读取输出来避免I/O死锁，
    同时保留了通过progress.json文件进行状态监控的能力。
    """
    if is_gpu_intensive:
        command = f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {command}"

    progress_log_file = None
    if use_progress_log:
        # 创建一个唯一的进度日志文件名
        progress_log_file = os.path.join(config.ITERATIVE_OUTPUT_DIR, f"progress_{int(time.time())}_{os.getpid()}.json")
        command = f"{command} --progress_log_file {progress_log_file}"
    
    logger.info(f"即将执行 [{description}]...")
    logger.debug(f"完整命令:\n---\n{command.strip()}\n---")
    
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding='utf-8', cwd=config.PROJECT_ROOT
    )

    try:
        # 监控阶段：只通过文件系统监控进度，不接触I/O管道
        if use_progress_log and progress_log_file:
            last_message = ""
            print() # 换行以开始新的进度条
            while process.poll() is None:
                if os.path.exists(progress_log_file):
                    try:
                        with open(progress_log_file, 'r', encoding='utf-8') as f:
                            status = json.load(f)
                        message = f"    └── 状态: {status['message']}"
                        if message != last_message:
                            # 使用 \r 和 ANSI 清除行来实现动态更新
                            print(f"\r\033[K{message}", end="", flush=True)
                            last_message = message
                    except (json.JSONDecodeError, FileNotFoundError, KeyError):
                        # 文件可能正在被写入或已被删除，忽略这些瞬时错误
                        pass
                time.sleep(1) # 降低轮询频率
            print() # 进度条结束后换行

        # 通信阶段：等待进程结束，然后一次性、安全地读取所有输出
        stdout, stderr = process.communicate()

        # 日志记录阶段
        if stdout:
            logger.info(f"--- 子进程 STDOUT [{description}] ---")
            for line in stdout.strip().split('\n'):
                logger.info(line)
        
        if process.returncode != 0:
            logger.error(f"命令 [{description}] 执行失败，返回码: {process.returncode}。")
            if stderr:
                logger.error(f"--- 子进程 STDERR [{description}] ---")
                for line in stderr.strip().split('\n'):
                    logger.error(line)
            raise subprocess.CalledProcessError(process.returncode, command)
        
        # 如果有stderr但返回码为0，通常是警告信息（例如tqdm进度条）
        if stderr:
            logger.debug(f"--- 子进程 STDERR (警告/调试信息) [{description}] ---")
            for line in stderr.strip().split('\n'):
                logger.debug(line)

        logger.info(f"✅ 命令 [{description}] 执行成功。")

    finally:
        # 清理进度文件
        if use_progress_log and progress_log_file and os.path.exists(progress_log_file):
            try:
                os.remove(progress_log_file)
            except OSError:
                pass

# --- 为了兼容性，保留旧的函数名，但都指向新的、健壮的实现 ---
def _run_command(command: str, description: str, is_gpu_intensive: bool = True):
    return _run_command_final(command, description, is_gpu_intensive, use_progress_log=False)

def _run_command_with_progress(command: str, description: str):
    return _run_command_final(command, description, is_gpu_intensive=True, use_progress_log=True)
# --- 【核心修复结束】 ---


def _verify_tensor_file(path: str, expected_dims: int, num_samples: int):
    logger.info(f"正在验证张量文件: {path}")
    if not os.path.exists(path): raise FileNotFoundError(f"致命错误：预期的输出文件未找到: {path}")
    tensor = torch.load(path)
    if not isinstance(tensor, torch.Tensor): raise TypeError(f"致命类型错误：文件 {path} 内容不是一个 PyTorch 张量。")
    if tensor.dim() != expected_dims: raise ValueError(f"致命维度错误：文件 {path} 中的张量维度为 {tensor.dim()}，应为 {expected_dims}。")
    if tensor.shape[0] != num_samples: raise ValueError(f"致命样本数错误：文件 {path} 中有 {tensor.shape[0]} 个样本，但预期有 {num_samples} 个。")
    if torch.isnan(tensor).any() or torch.isinf(tensor).any(): raise ValueError(f"致命数值错误：文件 {path} 中包含 NaN 或 Inf 值！")
    logger.info(f"✅ 文件 {path} 验证通过。张量形状: {tensor.shape}, 数值有效。")
    return tensor

def verify_optimizer_health(model_path: str, base_model_path: str):
    # ... (此函数及以下所有函数保持不变) ...
    optimizer_file = os.path.join(model_path, "optimizer.pt")
    logger.info(f"🔬 开始对优化器文件进行三级健康检查: {optimizer_file}")
    logger.info("--- [检查 1/3] 结构完整性...")
    if not os.path.exists(optimizer_file): raise FileNotFoundError(f"优化器文件不存在: {optimizer_file}")
    try: state_dict = torch.load(optimizer_file, map_location="cpu")
    except Exception as e: raise IOError(f"优化器文件加载失败: {e}")
    if "state" in state_dict: state = state_dict["state"]
    elif all(isinstance(k, int) for k in state_dict.keys()): state = state_dict
    else: raise ValueError("优化器文件中既没有 'state' 键，也不是一个有效的状态字典。")
    if not state: raise ValueError("优化器状态字典为空！")
    first_state_entry = list(state.values())[0]
    if 'exp_avg' not in first_state_entry or 'exp_avg_sq' not in first_state_entry: raise ValueError("状态条目中缺少 'exp_avg' 或 'exp_avg_sq' 键。")
    logger.info("✅ 结构检查通过。")
    logger.info("--- [检查 2/3] 数值有效性...")
    all_m_means, all_v_means = [], []
    for i, s in enumerate(state.values()):
        m, v = s['exp_avg'], s['exp_avg_sq']
        if torch.isnan(m).any() or torch.isinf(m).any() or torch.isnan(v).any() or torch.isinf(v).any(): raise ValueError(f"第 {i} 个参数的状态包含 NaN 或 Inf！")
        if (v < 0).any(): raise ValueError(f"第 {i} 个参数的二阶矩(v)包含负数！")
        all_m_means.append(m.abs().mean().item())
        all_v_means.append(v.abs().mean().item())
    avg_m_mean = sum(all_m_means)/len(all_m_means) if all_m_means else 0
    avg_v_mean = sum(all_v_means)/len(all_v_means) if all_v_means else 0
    if avg_m_mean < 1e-9 and avg_v_mean < 1e-9: raise ValueError("所有状态值几乎均为零！")
    logger.info(f"✅ 数值检查通过。平均m: {avg_m_mean:.2e}, 平均v: {avg_v_mean:.2e}")
    logger.info("--- [检查 3/3] 与模型结构匹配...")
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model = PeftModel.from_pretrained(model, model_path, is_trainable=True)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if len(state) != len(trainable_params): raise ValueError(f"优化器状态数量 ({len(state)}) 与模型可训练参数数量 ({len(trainable_params)}) 不匹配！")
        for i, p in enumerate(trainable_params):
            param_state = list(state.values())[i]
            if p.shape != param_state['exp_avg'].shape: raise ValueError(f"第 {i} 个参数的形状 ({p.shape}) 与其优化器状态的形状 ({param_state['exp_avg'].shape}) 不匹配！")
        logger.info("✅ 模型匹配检查通过。")
    except Exception as e: raise RuntimeError(f"加载模型或匹配参数时出错: {e}")
    logger.info("🎉 优化器文件健康检查完毕：健康、有效且与模型完全匹配！")

def calculate_gradients(model_path: str, data_file: str, num_samples: int, output_path: str, grad_type: str, is_lora: bool, base_model_path: str) -> str:
    if os.path.exists(output_path): shutil.rmtree(output_path)
    cmd_parts = ["python -m externals.less.data_selection.get_info", f"--model_path {model_path}", f"--base_model_path {base_model_path}", f"--train_file {data_file}", f"--output_path {output_path}", "--info_type grads", f"--gradient_type {grad_type}", f"--gradient_projection_dimension {config.GRADIENT_PROJECTION_DIM}"]
    cmd = " \\\n    ".join(cmd_parts)
    _run_command_with_progress(cmd, description=f"计算 {os.path.basename(data_file)} 的梯度")
    expected_grad_file = os.path.join(output_path, f"dim{config.GRADIENT_PROJECTION_DIM}", "all_unnormalized.pt")
    _verify_tensor_file(expected_grad_file, expected_dims=2, num_samples=num_samples)
    return expected_grad_file

def calculate_mmlu_validation_gradients(model_path: str, output_path: str, is_lora: bool, base_model_path: str) -> str:
    if os.path.exists(output_path): shutil.rmtree(output_path)
    cmd_parts = ["python -m externals.less.data_selection.get_info", f"--model_path {model_path}", f"--base_model_path {base_model_path}", "--task mmlu", f"--data_dir {config.DATA_DIR}", f"--output_path {output_path}", "--info_type grads", f"--gradient_type {config.GRADIENT_TYPE_VALIDATION}", f"--gradient_projection_dimension {config.GRADIENT_PROJECTION_DIM}"]
    cmd = " \\\n    ".join(cmd_parts)
    _run_command_with_progress(cmd, description="计算 MMLU 验证集梯度")
    expected_samples = 57 * 5
    expected_grad_file = os.path.join(output_path, f"dim{config.GRADIENT_PROJECTION_DIM}", "all_unnormalized.pt")
    _verify_tensor_file(expected_grad_file, expected_dims=2, num_samples=expected_samples)
    return expected_grad_file

def run_finetuning(base_model_name: str, train_file: str, output_dir: str, num_train_samples: int) -> str:
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    sft_args = config.LORA_TRAIN_ARGS; args_list = [f"--model_name_or_path {base_model_name}", f"--train_file {train_file}", f"--output_dir {output_dir}", f"--seed {config.RANDOM_SEED}"]
    for k, v in sft_args.items():
        if isinstance(v, bool) and v: args_list.append(f"--{k}")
        elif not isinstance(v, bool): args_list.append(f"--{k} {v}")
    args_str = " ".join(args_list); cmd = f"python -m finetune {args_str}"
    _run_command(cmd, description=f"在 {num_train_samples} 条数据上微调模型", is_gpu_intensive=True)
    if not os.path.exists(os.path.join(output_dir, "adapter_config.json")):
        raise FileNotFoundError(f"微调结束后，预期的 LoRA 配置文件未在 {output_dir} 中找到。")
    return output_dir

def run_evaluation(model_path: str, eval_output_dir: str, eval_split: str) -> tuple[float, float]:
    split_output_dir = os.path.join(eval_output_dir, eval_split);
    if os.path.exists(split_output_dir): shutil.rmtree(split_output_dir); os.makedirs(split_output_dir)
    eval_args_str = " ".join(f"--{k} {v}" if not isinstance(v, bool) else (f"--{k}" if v else "") for k, v in config.EVAL_ARGS.items())
    split_arg = "--eval_valid" if eval_split == "dev" else ""; base_model_arg = f"--base_model_path {config.BASE_MODEL_NAME}"
    cmd = (f"python -m externals.evaluation.run_eval {eval_args_str} --model_name_or_path {model_path} --data_dir {config.MMLU_DATA_DIR} --save_dir {split_output_dir} {split_arg} {base_model_arg}")
    _run_command(cmd, description=f"在 MMLU {eval_split} 集上评估")
    metrics_file = os.path.join(split_output_dir, "metrics.json")
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"评估结束后，结果文件 {metrics_file} 未找到。")
    try:
        with open(metrics_file, 'r') as f: metrics = json.load(f)
        accuracy = metrics.get("average_acc", -1.0); loss = metrics.get("average_loss", -1.0)
        logger.info(f"✅ 评估成功。Acc: {accuracy:.4f}, Loss: {loss:.4f}"); return accuracy, loss
    except (json.JSONDecodeError, KeyError) as e: raise RuntimeError(f"解析评估结果文件 {metrics_file} 时出错: {e}")

def run_matching_subprocess(train_grad_path: str, val_grad_path: str, selection_size: int, num_train_samples: int, phase_dir: str) -> tuple[list[int], pd.DataFrame]:
    output_dir = os.path.join(phase_dir, "matching_results")
    os.makedirs(output_dir, exist_ok=True)
    output_indices_path = os.path.join(output_dir, "top_indices.pt")
    output_scores_path = os.path.join(output_dir, "all_scores.jsonl")
    cmd_parts = ["python", "run_matching.py", f"--train_grad_path {train_grad_path}", f"--val_grad_path {val_grad_path}", f"--num_train_samples {num_train_samples}", f"--selection_size {selection_size}", f"--output_indices_path {output_indices_path}", f"--output_scores_path {output_scores_path}"]
    cmd = " ".join(cmd_parts)
    _run_command(cmd, description="执行分数计算子进程")
    if not os.path.exists(output_indices_path) or not os.path.exists(output_scores_path):
        raise RuntimeError("分数计算子进程未能成功生成输出文件！")
    top_k_indices = torch.load(output_indices_path).tolist()
    scores_df = pd.read_json(output_scores_path, lines=True)
    logger.info(f"✅ 成功从子进程加载分数和索引。选择了 {len(top_k_indices)} 个索引。")
    return top_k_indices, scores_df