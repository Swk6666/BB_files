#!/usr/bin/env python3
"""
测试脚本：检查实际优化器状态的精度
"""
import torch
import os

def check_optimizer_precision():
    """检查您的实际优化器文件中的精度"""
    
    # 检查您的优化器文件
    optimizer_path = "/root/autodl-tmp/results/IterativeLess_Llama-7B_Final_Run_Robust/warmup_model/optimizer.pt"
    
    if not os.path.exists(optimizer_path):
        print(f"优化器文件不存在: {optimizer_path}")
        return
    
    print("正在加载优化器状态...")
    optimizer_state_dict = torch.load(optimizer_path, map_location="cpu")
    
    print("优化器状态字典的键:")
    for key in optimizer_state_dict.keys():
        print(f"  - {key}: {type(optimizer_state_dict[key])}")
    
    if "state" in optimizer_state_dict:
        state = optimizer_state_dict["state"]
        print(f"\n'state' 包含 {len(state)} 个参数的状态")
        
        # 检查前几个状态的精度
        for i, (param_id, param_state) in enumerate(list(state.items())[:3]):
            print(f"\n参数 {param_id} 的状态:")
            for state_key, state_value in param_state.items():
                if isinstance(state_value, torch.Tensor):
                    print(f"  - {state_key}: 形状={state_value.shape}, 精度={state_value.dtype}, 设备={state_value.device}")
                else:
                    print(f"  - {state_key}: {type(state_value)} = {state_value}")
            
            if i >= 2:  # 只检查前3个
                break
    
    elif isinstance(optimizer_state_dict, dict) and all(isinstance(k, int) for k in optimizer_state_dict.keys()):
        print(f"优化器状态直接是参数字典，包含 {len(optimizer_state_dict)} 个参数")
        
        # 检查前几个参数状态的精度
        for i, (param_id, param_state) in enumerate(list(optimizer_state_dict.items())[:3]):
            print(f"\n参数 {param_id} 的状态:")
            for state_key, state_value in param_state.items():
                if isinstance(state_value, torch.Tensor):
                    print(f"  - {state_key}: 形状={state_value.shape}, 精度={state_value.dtype}, 设备={state_value.device}")
                else:
                    print(f"  - {state_key}: {type(state_value)} = {state_value}")
            
            if i >= 2:
                break
    
    else:
        print("无法识别优化器状态的结构")

def simulate_mixed_precision_adam():
    """模拟不同精度组合下的Adam计算"""
    print("\n" + "="*60)
    print("模拟不同精度组合下的Adam计算")
    print("="*60)
    
    # 测试不同的精度组合
    test_cases = [
        ("所有float32", torch.float32, torch.float32, torch.float32),
        ("所有bfloat16", torch.bfloat16, torch.bfloat16, torch.bfloat16),
        ("优化器float32+梯度bfloat16", torch.float32, torch.bfloat16, torch.float32),
        ("优化器bfloat16+梯度bfloat16", torch.bfloat16, torch.bfloat16, torch.bfloat16),
    ]
    
    for case_name, opt_dtype, grad_dtype, result_dtype in test_cases:
        print(f"\n--- {case_name} ---")
        
        # 创建测试数据
        m_global = torch.randn(1000, dtype=opt_dtype) * 0.1
        v_global = torch.rand(1000, dtype=opt_dtype) * 0.01 + 1e-8
        vectorized_grads = torch.randn(1000, dtype=grad_dtype)
        
        # Adam参数
        beta1, beta2, eps = 0.9, 0.999, 1e-08
        
        try:
            # 执行Adam计算
            updated_avg = beta1 * m_global + (1 - beta1) * vectorized_grads
            updated_avg_sq = beta2 * v_global + (1 - beta2) * vectorized_grads ** 2
            result = updated_avg / torch.sqrt(updated_avg_sq + eps)
            
            print(f"  计算成功")
            print(f"  结果精度: {result.dtype}")
            print(f"  是否包含NaN: {torch.isnan(result).any().item()}")
            print(f"  是否包含Inf: {torch.isinf(result).any().item()}")
            print(f"  结果范围: [{result.min().item():.2e}, {result.max().item():.2e}]")
            
        except Exception as e:
            print(f"  计算失败: {e}")

if __name__ == "__main__":
    check_optimizer_precision()
    simulate_mixed_precision_adam()
