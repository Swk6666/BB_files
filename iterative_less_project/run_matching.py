# /root/iterative_less_project/run_matching.py
import argparse
import os
import torch
import pandas as pd
import torch.nn.functional as F
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_tensor_file(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"输入文件未找到: {path}")
    tensor = torch.load(path)
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(f"输入文件 {path} 包含 NaN 或 Inf 值！")
    return tensor

def main():
    parser = argparse.ArgumentParser(description="独立的、隔离的分数计算脚本。")
    parser.add_argument("--train_grad_path", type=str, required=True)
    parser.add_argument("--val_grad_path", type=str, required=True)
    parser.add_argument("--num_train_samples", type=int, required=True)
    parser.add_argument("--selection_size", type=int, required=True)
    parser.add_argument("--output_indices_path", type=str, required=True)
    parser.add_argument("--output_scores_path", type=str, required=True)
    args = parser.parse_args()

    logger.info("在独立的子进程中开始计算影响力分数...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    try:
        train_grads = verify_tensor_file(args.train_grad_path).to(device)
        val_grads = verify_tensor_file(args.val_grad_path).to(device)
        assert train_grads.shape[0] == args.num_train_samples

        train_grads_norm = F.normalize(train_grads.float(), p=2, dim=1)
        val_grads_norm = F.normalize(val_grads.float(), p=2, dim=1)
        
        scores_df = pd.DataFrame({'score': [0.0] * args.num_train_samples})
        
        if torch.isnan(train_grads_norm).any():
            logger.error("FATAL: 训练集梯度在归一化后出现NaN，可能包含全零向量。")
            valid_rows = ~torch.isnan(train_grads_norm).any(dim=1)
            if valid_rows.any():
                logger.info(f"尝试从 {valid_rows.sum().item()} 个有效行中恢复分数计算...")
                similarity_matrix = torch.matmul(train_grads_norm[valid_rows], val_grads_norm.T)
                final_scores = similarity_matrix.max(dim=1)[0]
                scores_df.loc[valid_rows.cpu(), 'score'] = final_scores.cpu().numpy()
        else:
            similarity_matrix = torch.matmul(train_grads_norm, val_grads_norm.T)
            final_scores = similarity_matrix.max(dim=1)[0]
            scores_df['score'] = final_scores.cpu().numpy()

        k = min(args.selection_size, len(scores_df))
        top_k_indices = torch.topk(torch.from_numpy(scores_df['score'].values), k=k)[1]

        # 保存结果到文件
        scores_df.to_json(args.output_scores_path, orient="records", lines=True)
        torch.save(top_k_indices.cpu(), args.output_indices_path)
        
        logger.info(f"✅ 分数计算成功，结果已保存到:")
        logger.info(f"   - 分数: {args.output_scores_path}")
        logger.info(f"   - 索引: {args.output_indices_path}")

    except Exception as e:
        logger.critical("分数计算子进程发生致命错误！", exc_info=True)
        # 创建空的失败标记文件
        open(args.output_scores_path + ".error", 'w').close()
        open(args.output_indices_path + ".error", 'w').close()
        raise e
    finally:
        del train_grads, val_grads, train_grads_norm, val_grads_norm
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()