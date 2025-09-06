# /root/iterative_less_project/externals/less/data_selection/collect_grad_reps.py
import os
import logging
from typing import List, Optional, Iterable
import math
import torch
from peft import PeftModel
from tqdm import tqdm
from trak.projectors import CudaProjector, ProjectionType
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def get_number_of_params(model):
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f"模型可训练参数数量: {num_params}")
    return num_params

def collect_grads(data_iterable: Iterable, model, output_dir: str, proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None, gradient_type: str = "adam",
                  max_samples: Optional[int] = None):
    model_id = 0
    torch.random.manual_seed(0)
    
    CHUNK_SIZE = 8
    device = next(model.parameters()).device
    number_of_params = get_number_of_params(model)
    
    m_global, v_global = None, None
    if gradient_type == "adam":
        if adam_optimizer_state is None: raise ValueError("Adam influence computation requires a valid adam_optimizer_state.")
        logger.info("正在准备有状态Adam影响力的全局 m 和 v 向量...")
        params_with_grad = [p for p in model.parameters() if p.requires_grad]
        state_values = list(adam_optimizer_state.values())
        if len(state_values) != len(params_with_grad): raise RuntimeError(f"优化器状态数量 ({len(state_values)}) 与模型可训练参数数量 ({len(params_with_grad)}) 不匹配!")
        m_list = [state_values[i]['exp_avg'].flatten().to(device, dtype=torch.float32) for i in range(len(params_with_grad))]
        v_list = [state_values[i]['exp_avg_sq'].flatten().to(device, dtype=torch.float32) for i in range(len(params_with_grad))]
        m_global = torch.cat(m_list)
        v_global = torch.cat(v_list)
        if m_global.numel() != number_of_params or v_global.numel() != number_of_params: raise RuntimeError("Reconstructed Adam moment vectors' size does not match model's trainable parameters.")
        logger.info(f"✅ 全局 m 和 v 向量准备完毕。")

    projectors = {dim: CudaProjector(grad_dim=number_of_params, proj_dim=dim, seed=0, device=device, dtype=torch.float32, proj_type=ProjectionType.rademacher, max_batch_size=CHUNK_SIZE) for dim in proj_dim}

    all_projected_grads = {dim: [] for dim in proj_dim}
    full_grads_cpu_buffer = []
    model.train()
    
    count = 0
    skipped_samples = 0
    total_samples = len(data_iterable) if hasattr(data_iterable, '__len__') else -1
    if isinstance(data_iterable, DataLoader): total_samples = len(data_iterable.dataset) # 对DataLoader获取总样本数
    if max_samples is not None: total_samples = min(max_samples, total_samples) if total_samples != -1 else max_samples

    # --- 【核心修复】 ---
    # 循环的单位是“批次”(batch)，而不是“样本”(sample)
    for batch in tqdm(data_iterable, desc="Computing Gradients", total=len(data_iterable)):
        if max_samples is not None and count >= max_samples: break
        
        # 兼容两种数据源：
        # 1. DataLoader: batch 是一个字典，其值是形状为 [batch_size, seq_len] 的Tensor
        # 2. 预加载列表: batch 是一个字典，其值是形状为 [seq_len] 的Tensor
        is_from_dataloader = isinstance(data_iterable, DataLoader)
        
        # 如果不是来自DataLoader，我们需要手动创建batch维度
        if not is_from_dataloader:
            temp_batch = {}
            for key, tensor in batch.items():
                 # 防御性检查
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"数据格式错误！期望得到Tensor，但对于键'{key}'却得到了{type(tensor)}。这不应该发生。")
                temp_batch[key] = tensor.unsqueeze(0).to(device)
            batch = temp_batch
        else: # 如果来自DataLoader，数据已经是带batch维度的Tensor，直接送到device即可
            batch = {k: v.to(device) for k, v in batch.items()}

        count += batch['input_ids'].shape[0] # 按实际batch大小增加计数
        # --- 【修复结束】 ---

        model.zero_grad()
        loss = model(**batch).loss

        if torch.isinf(loss) or torch.isnan(loss):
            logger.warning(f"样本 {count} 损失值为 {loss.item()}，已跳过。")
            skipped_samples += 1
            continue

        loss.backward()

        with torch.no_grad():
            vectorized_grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.requires_grad])

        if torch.isinf(vectorized_grads).any() or torch.isnan(vectorized_grads).any():
            logger.warning(f"样本 {count} 的原始梯度包含 NaN/Inf，已跳过。")
            skipped_samples += 1
            continue

        if gradient_type == "adam":
            beta1, beta2, eps = 0.9, 0.999, 1e-8
            g_fp32 = vectorized_grads.float()
            m_t = m_global * beta1 + g_fp32 * (1 - beta1)
            v_t = v_global * beta2 + g_fp32.pow(2) * (1 - beta2)
            denom = v_t.sqrt()
            denom = torch.nan_to_num(denom, nan=0.0, posinf=1e9)
            denom += eps
            adam_influence_grads = m_t / denom
            adam_influence_grads = torch.nan_to_num(adam_influence_grads, nan=0.0, posinf=0.0, neginf=0.0)
            vectorized_grads = adam_influence_grads.to(model.dtype)
        
        full_grads_cpu_buffer.append(vectorized_grads.cpu())

        if len(full_grads_cpu_buffer) == CHUNK_SIZE or (count >= total_samples and full_grads_cpu_buffer):
            grads_tensor_chunk = torch.stack(full_grads_cpu_buffer).to(torch.float32)
            for dim, projector in projectors.items():
                projected_chunk = projector.project(grads_tensor_chunk.to(device), model_id=model_id)
                all_projected_grads[dim].append(projected_chunk.cpu().to(torch.bfloat16))
            full_grads_cpu_buffer = []
        
    torch.cuda.empty_cache()
    
    logger.info(f"梯度计算循环完成。总样本数: {count}, 跳过样本数: {skipped_samples}")
    for dim in proj_dim:
        if not all_projected_grads[dim]:
            if skipped_samples == count and count > 0:
                raise RuntimeError(f"致命错误: 所有 {count} 个样本都被跳过，未能生成任何有效的梯度！")
            else:
                logger.warning(f"未能为维度 {dim} 收集任何梯度。")
                continue
        merged_data = torch.cat(all_projected_grads[dim])
        if torch.isinf(merged_data).any() or torch.isnan(merged_data).any():
            inf_count = torch.isinf(merged_data).sum().item()
            nan_count = torch.isnan(merged_data).sum().item()
            raise RuntimeError(f"致命错误: 最终保存的梯度文件 dim{dim} 中仍然包含 NaN/Inf！(Inf: {inf_count}, NaN: {nan_count})")
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        os.makedirs(output_dir_per_dim, exist_ok=True)
        output_file = os.path.join(output_dir_per_dim, f"all_unnormalized.pt")
        torch.save(merged_data, output_file)
        logger.info(f"✅ 已成功保存最终的未归一化影响力梯度 (形状: {merged_data.shape}) 到 {output_file}")

    logger.info("✅ 梯度收集和投影完成。")