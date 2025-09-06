# /root/iterative_less_project/debug_gradients.py
import torch
import os
import logging
import torch.nn.functional as F
from iterative_less import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')

def analyze_gradients(tensor: torch.Tensor, name: str):
    """对梯度张量进行详细的数值分析"""
    logging.info(f"\n--- Analyzing Gradients for: {name} ---")
    if not isinstance(tensor, torch.Tensor):
        logging.error("Input is not a torch.Tensor!")
        return

    # 检查基本属性
    logging.info(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}, Device: {tensor.device}")

    # 检查是否有 NaN 或 Inf
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    logging.info(f"Contains NaN: {has_nan.item()}")
    logging.info(f"Contains Inf: {has_inf.item()}")

    if has_nan or has_inf:
        logging.error("Tensor contains invalid values (NaN/Inf), stopping further analysis.")
        return

    # 计算每个梯度向量的 L2 范数（模长）
    norms = torch.linalg.norm(tensor.float(), dim=1)
    
    # 打印范数的统计信息
    logging.info("Statistics of Vector L2 Norms (Magnitude):")
    logging.info(f"  - Mean:   {norms.mean().item():.6e}")
    logging.info(f"  - Std:    {norms.std().item():.6e}")
    logging.info(f"  - Min:    {norms.min().item():.6e}")
    logging.info(f"  - Max:    {norms.max().item():.6e}")
    logging.info(f"  - Num zeros: {torch.sum(norms == 0).item()} / {len(norms)}")

    # 打印张量元素本身的统计信息
    logging.info("Statistics of Individual Gradient Values:")
    logging.info(f"  - Mean:   {tensor.float().mean().item():.6e}")
    logging.info(f"  - Std:    {tensor.float().std().item():.6e}")
    logging.info(f"  - Min:    {tensor.float().min().item():.6e}")
    logging.info(f"  - Max:    {tensor.float().max().item():.6e}")

def main():
    # 指向您实验输出的梯度文件
    train_grad_path = os.path.join(config.ITERATIVE_OUTPUT_DIR, "one_shot_gradients/grads_train_pool/dim8192/all_unnormalized.pt")
    val_grad_path = os.path.join(config.ITERATIVE_OUTPUT_DIR, "one_shot_gradients/grads_validation/dim8192/all_unnormalized.pt")

    if not os.path.exists(train_grad_path) or not os.path.exists(val_grad_path):
        logging.error("Gradient files not found! Please run the main experiment first.")
        logging.error(f"Expected train grads at: {train_grad_path}")
        logging.error(f"Expected val grads at:   {val_grad_path}")
        return

    # 加载梯度
    train_grads = torch.load(train_grad_path, map_location="cpu")
    val_grads = torch.load(val_grad_path, map_location="cpu")

    # 分析两个梯度张量
    analyze_gradients(train_grads, "Candidate Pool (Adam Gradients)")
    analyze_gradients(val_grads, "MMLU Validation (SGD Gradients)")

    # 模拟相似度计算
    logging.info("\n--- Simulating Similarity Calculation ---")
    
    # 检查归一化后的范数
    train_grads_norm = F.normalize(train_grads.float(), p=2, dim=1)
    val_grads_norm = F.normalize(val_grads.float(), p=2, dim=1)
    logging.info("L2 Norm of normalized train grads (should be 1.0):")
    logging.info(torch.linalg.norm(train_grads_norm, dim=1).describe())
    logging.info("L2 Norm of normalized val grads (should be 1.0):")
    logging.info(torch.linalg.norm(val_grads_norm, dim=1).describe())

    # 计算相似度矩阵
    similarity_matrix = torch.matmul(train_grads_norm, val_grads_norm.T)
    logging.info(f"Similarity Matrix Shape: {similarity_matrix.shape}")
    
    # 计算最终得分
    final_scores = similarity_matrix.max(dim=1)[0]
    logging.info("Statistics of Final Scores:")
    logging.info(f"  - Mean:   {final_scores.mean().item():.6f}")
    logging.info(f"  - Std:    {final_scores.std().item():.6f}")
    logging.info(f"  - Min:    {final_scores.min().item():.6f}")
    logging.info(f"  - Max:    {final_scores.max().item():.6f}")
    logging.info(f"  - Num zeros: {torch.sum(final_scores == 0).item()} / {len(final_scores)}")

if __name__ == "__main__":
    main()