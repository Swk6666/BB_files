# /root/iterative_less_project/isolate_bad_sample.py
import argparse
import os
import torch
import logging
from transformers import AutoTokenizer

# 动态添加项目路径
import sys
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入我们自己的模块
from iterative_less.progress_logger import ProgressLogger
from externals.less.data_selection.collect_grad_reps import collect_grads
from externals.less.data_selection.get_training_dataset import get_training_dataset
from externals.less.data_selection.get_info import load_model # 复用模型加载函数
from iterative_less import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def run_test_on_slice(start_index: int, end_index: int):
    """
    只对 candidate_pool.jsonl 的一个切片运行梯度计算。
    """
    logger.info(f"--- 开始对索引范围 [{start_index}, {end_index}) 的数据进行隔离测试 ---")
    
    # 构造一个临时的进度日志文件
    progress_log_file = f"/tmp/isolate_progress_{start_index}_{end_index}.json"
    progress = ProgressLogger(progress_log_file)
    
    try:
        # 1. 加载模型
        progress.log("Loading model...")
        model_path = os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_model")
        model = load_model(model_path, config.BASE_MODEL_NAME, progress)
        
        # 2. 加载并切片数据
        progress.log("Loading and slicing data...")
        tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME, use_fast=True, local_files_only=True)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        
        full_dataset = get_training_dataset(config.CANDIDATE_POOL_FILENAME, tokenizer, 1024, num_workers=1)
        
        if start_index >= len(full_dataset) or end_index > len(full_dataset):
            logger.error(f"索引范围错误！请求范围 [{start_index}, {end_index})，但数据集大小为 {len(full_dataset)}。")
            return

        # 将切片数据预加载到内存
        data_slice = [full_dataset[i] for i in range(start_index, end_index)]
        logger.info(f"已将 {len(data_slice)} 个样本加载到内存，准备计算梯度。")

        # 3. 加载优化器状态 (与主流程保持一致)
        progress.log("Loading optimizer state...")
        optimizer_path = os.path.join(model_path, "optimizer.pt")
        optimizer_state_dict = torch.load(optimizer_path, map_location="cpu")
        adam_param_states = optimizer_state_dict.get("state", optimizer_state_dict)

        # 4. 运行梯度计算
        progress.log("Starting gradient computation on slice...")
        output_dir = f"/tmp/isolate_output_{start_index}_{end_index}"
        
        collect_grads(
            data_iterable=data_slice,
            model=model,
            output_dir=output_dir,
            proj_dim=[config.GRADIENT_PROJECTION_DIM],
            gradient_type=config.GRADIENT_TYPE_TRAIN_POOL_ADAM,
            adam_optimizer_state=adam_param_states
        )
        
        logger.info(f"✅ 成功完成索引范围 [{start_index}, {end_index}) 的测试！")
        
    except Exception as e:
        logger.critical(f"在处理索引范围 [{start_index}, {end_index}) 时发生致命错误！", exc_info=True)
        # 打印出这个范围内的原始索引，方便追查
        import pandas as pd
        df = pd.read_json(config.CANDIDATE_POOL_FILENAME, lines=True)
        problematic_indices = df.iloc[start_index:end_index]['original_index'].tolist()
        logger.error(f"发生错误的样本可能位于以下 original_index 中: {problematic_indices}")

    finally:
        # 清理临时文件
        import shutil
        if os.path.exists(f"/tmp/isolate_output_{start_index}_{end_index}"):
            shutil.rmtree(f"/tmp/isolate_output_{start_index}_{end_index}")
        if os.path.exists(progress_log_file):
            os.remove(progress_log_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="隔离测试脚本，用于定位导致挂起的坏数据。")
    parser.add_argument("--start", type=int, required=True, help="起始索引 (包含)")
    parser.add_argument("--end", type=int, required=True, help="结束索引 (不包含)")
    args = parser.parse_args()
    
    run_test_on_slice(args.start, args.end)