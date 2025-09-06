# /root/iterative_less_project/run_less_scoring.py
import argparse
import os
import torch
import pandas as pd
import torch.nn.functional as F
import shutil
import sys
import logging

# 将项目根目录添加到sys.path，以便能导入iterative_less模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from iterative_less import config
# 为避免循环导入，我们在此处定义一个最小化的命令执行器
from iterative_less.toolkit import _run_command, _verify_tensor_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_gradients_standalone(model_path: str, data_file: str, num_samples: int, output_path: str, grad_type: str, is_lora: bool, base_model_path: str) -> str:
    """一个独立的梯度计算调用函数，避免循环导入。"""
    logger.info(f"开始为 {num_samples} 条数据在 '{os.path.basename(data_file)}' 上计算梯度...")
    if os.path.exists(output_path): shutil.rmtree(output_path)
    cmd_parts = ["python -m externals.less.data_selection.get_info", f"--model_path {model_path}", f"--base_model_path {base_model_path}", f"--train_file {data_file}", f"--output_path {output_path}", "--info_type grads", f"--gradient_type {grad_type}", f"--gradient_projection_dimension {config.GRADIENT_PROJECTION_DIM}"]
    if not is_lora: cmd_parts.append("--initialize_lora")
    cmd = " \\\n    ".join(cmd_parts)
    _run_command(cmd)
    # 加载未经归一化的原始梯度文件
    expected_grad_file = os.path.join(output_path, f"dim{config.GRADIENT_PROJECTION_DIM}", "all_unnormalized.pt")
    _verify_tensor_file(expected_grad_file, expected_dims=2, num_samples=num_samples)
    logger.info(f"梯度计算成功。"); return expected_grad_file

def main():
    parser = argparse.ArgumentParser(description="独立的LESS评分脚本，确保内存隔离。")
    parser.add_argument("--llama_model_path", type=str, required=True)
    parser.add_argument("--val_grad_path", type=str, required=True)
    parser.add_argument("--input_df_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_scores_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_json(args.input_df_path, lines=True)
    num_samples = len(df)
    
    if num_samples == 0:
        pd.DataFrame(columns=["original_index", "less_score"]).to_json(args.output_scores_path, orient='records', lines=True)
        logger.info("输入数据为空，已生成空的分数文件。")
        return

    temp_less_input_file = os.path.join(args.output_dir, "temp_input_for_less.jsonl")
    df_to_save = df[['original_index', 'imputed_text']].copy()
    df_to_save.rename(columns={'imputed_text': 'text'}, inplace=True)
    df_to_save.to_json(temp_less_input_file, orient='records', lines=True, force_ascii=False)

    train_grad_output_path = os.path.join(args.output_dir, "temp_grads")
    train_grad_path = calculate_gradients_standalone(
        model_path=args.llama_model_path,
        data_file=temp_less_input_file,
        num_samples=num_samples,
        output_path=train_grad_output_path,
        # 候选数据池梯度必须使用 Adam 影响力梯度
        grad_type="adam",
        is_lora=True,
        base_model_path=config.BASE_MODEL_NAME
    )
    
    train_grads = torch.load(train_grad_path, map_location=device)
    val_grads = torch.load(args.val_grad_path, map_location=device)
    
    train_grads_normalized = F.normalize(train_grads, p=2, dim=1)
    val_grads_normalized = F.normalize(val_grads, p=2, dim=1)
    influence_scores = torch.matmul(train_grads_normalized, val_grads_normalized.T)

    final_scores = influence_scores.max(dim=1)[0].cpu().numpy()
    scores_df = pd.DataFrame({"original_index": df['original_index'], "less_score": final_scores})
    
    scores_df.to_json(args.output_scores_path, orient='records', lines=True)
    logger.info(f"LESS分数计算完成，结果已保存到: {args.output_scores_path}")

    del train_grads, val_grads, train_grads_normalized, val_grads_normalized, influence_scores
    torch.cuda.empty_cache()
    shutil.rmtree(train_grad_output_path, ignore_errors=True)
    os.remove(temp_less_input_file)

if __name__ == "__main__":
    main()
