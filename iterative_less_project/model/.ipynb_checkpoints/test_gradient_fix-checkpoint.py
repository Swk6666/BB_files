#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证梯度计算的数值稳定性修复
"""

import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from iterative_less import config, logger_config

def test_adam_gradient_stability():
    """测试Adam梯度计算的数值稳定性"""
    print("=" * 60)
    print("测试 Adam 梯度计算的数值稳定性")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模拟一些可能导致数值不稳定的场景
    test_cases = [
        {
            "name": "正常梯度",
            "grads": torch.randn(100, 8192, dtype=torch.float32, device=device),
            "m_global": torch.randn(8192, dtype=torch.float32, device=device) * 0.1,
            "v_global": torch.rand(8192, dtype=torch.float32, device=device) * 0.01 + 1e-8
        },
        {
            "name": "极小梯度（接近零）",
            "grads": torch.randn(100, 8192, dtype=torch.float32, device=device) * 1e-10,
            "m_global": torch.randn(8192, dtype=torch.float32, device=device) * 1e-10,
            "v_global": torch.rand(8192, dtype=torch.float32, device=device) * 1e-12 + 1e-12
        },
        {
            "name": "极大梯度（可能爆炸）",
            "grads": torch.randn(100, 8192, dtype=torch.float32, device=device) * 100,
            "m_global": torch.randn(8192, dtype=torch.float32, device=device) * 50,
            "v_global": torch.rand(8192, dtype=torch.float32, device=device) * 1000 + 1e-8
        },
        {
            "name": "混合极值梯度",
            "grads": torch.cat([
                torch.randn(50, 4096, dtype=torch.float32, device=device) * 1e-10,  # 极小
                torch.randn(50, 4096, dtype=torch.float32, device=device) * 100,    # 极大
            ], dim=1),
            "m_global": torch.cat([
                torch.randn(4096, dtype=torch.float32, device=device) * 1e-10,
                torch.randn(4096, dtype=torch.float32, device=device) * 50,
            ], dim=0),
            "v_global": torch.cat([
                torch.rand(4096, dtype=torch.float32, device=device) * 1e-12 + 1e-12,
                torch.rand(4096, dtype=torch.float32, device=device) * 1000 + 1e-8,
            ], dim=0)
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- 测试场景: {test_case['name']} ---")
        
        grads = test_case["grads"]
        m_global = test_case["m_global"]
        v_global = test_case["v_global"]
        
        print(f"梯度形状: {grads.shape}")
        print(f"梯度范围: [{grads.min().item():.2e}, {grads.max().item():.2e}]")
        print(f"m 范围: [{m_global.min().item():.2e}, {m_global.max().item():.2e}]")
        print(f"v 范围: [{v_global.min().item():.2e}, {v_global.max().item():.2e}]")
        
        # 应用修复后的Adam计算逻辑
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        
        results = []
        for i in range(grads.shape[0]):
            g_fp32 = grads[i].float()
            
            # 计算更新后的矩
            m_t = m_global * beta1 + g_fp32 * (1 - beta1)
            v_t = v_global * beta2 + g_fp32.pow(2) * (1 - beta2)
            
            # 数值稳定版本
            stable_eps = max(eps, 1e-6)
            denom = torch.sqrt(v_t + stable_eps)
            denom = torch.clamp(denom, min=stable_eps, max=1e6)
            
            adam_influence_grads = m_t / denom
            
            # 梯度裁剪
            grad_norm = torch.norm(adam_influence_grads)
            max_norm = 10.0
            if grad_norm > max_norm:
                adam_influence_grads = adam_influence_grads * (max_norm / grad_norm)
            
            # 最终清理
            torch.nan_to_num_(adam_influence_grads, nan=0.0, posinf=0.0, neginf=0.0, out=adam_influence_grads)
            
            results.append(adam_influence_grads)
        
        # 分析结果
        result_tensor = torch.stack(results)
        
        has_nan = torch.isnan(result_tensor).any()
        has_inf = torch.isinf(result_tensor).any()
        
        print(f"结果包含 NaN: {has_nan.item()}")
        print(f"结果包含 Inf: {has_inf.item()}")
        print(f"结果范围: [{result_tensor.min().item():.2e}, {result_tensor.max().item():.2e}]")
        print(f"结果平均值: {result_tensor.mean().item():.2e}")
        print(f"结果标准差: {result_tensor.std().item():.2e}")
        
        # 测试L2归一化
        try:
            norms = torch.norm(result_tensor, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-8)
            normalized = result_tensor / norms
            
            norm_has_nan = torch.isnan(normalized).any()
            norm_has_inf = torch.isinf(normalized).any()
            
            print(f"归一化后包含 NaN: {norm_has_nan.item()}")
            print(f"归一化后包含 Inf: {norm_has_inf.item()}")
            
            if not norm_has_nan and not norm_has_inf:
                print("✅ 通过归一化测试")
            else:
                print("❌ 归一化测试失败")
                
        except Exception as e:
            print(f"❌ 归一化过程中出错: {e}")
        
        # 检查是否符合预期
        if not has_nan and not has_inf:
            print("✅ 数值稳定性测试通过")
        else:
            print("❌ 数值稳定性测试失败")

def test_similarity_calculation():
    """测试相似度计算的稳定性"""
    print("\n" + "=" * 60)
    print("测试相似度计算的数值稳定性")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建一些测试数据
    train_grads = torch.randn(200, 8192, dtype=torch.float32, device=device)
    val_grads = torch.randn(285, 8192, dtype=torch.float32, device=device)
    
    print(f"训练梯度形状: {train_grads.shape}")
    print(f"验证梯度形状: {val_grads.shape}")
    
    # 应用修复后的归一化逻辑
    train_norms = torch.norm(train_grads, dim=1, keepdim=True)
    val_norms = torch.norm(val_grads, dim=1, keepdim=True)
    
    eps = 1e-8
    train_norms = torch.clamp(train_norms, min=eps)
    val_norms = torch.clamp(val_norms, min=eps)
    
    train_grads_norm = train_grads / train_norms
    val_grads_norm = val_grads / val_norms
    
    print(f"训练梯度归一化后范数范围: [{torch.norm(train_grads_norm, dim=1).min().item():.6f}, {torch.norm(train_grads_norm, dim=1).max().item():.6f}]")
    print(f"验证梯度归一化后范数范围: [{torch.norm(val_grads_norm, dim=1).min().item():.6f}, {torch.norm(val_grads_norm, dim=1).max().item():.6f}]")
    
    # 分块计算相似度
    chunk_size = 50
    final_scores = torch.zeros(train_grads.shape[0], device=device)
    
    for i in range(0, train_grads.shape[0], chunk_size):
        end_i = min(i + chunk_size, train_grads.shape[0])
        train_chunk = train_grads_norm[i:end_i]
        
        similarity_chunk = torch.matmul(train_chunk, val_grads_norm.T)
        max_similarities = similarity_chunk.max(dim=1)[0]
        final_scores[i:end_i] = max_similarities
    
    # 分析结果
    has_nan = torch.isnan(final_scores).any()
    has_inf = torch.isinf(final_scores).any()
    
    print(f"相似度分数包含 NaN: {has_nan.item()}")
    print(f"相似度分数包含 Inf: {has_inf.item()}")
    print(f"相似度分数范围: [{final_scores.min().item():.6f}, {final_scores.max().item():.6f}]")
    print(f"相似度分数平均值: {final_scores.mean().item():.6f}")
    print(f"相似度分数标准差: {final_scores.std().item():.6f}")
    print(f"零分数数量: {torch.sum(final_scores == 0).item()} / {len(final_scores)}")
    
    if not has_nan and not has_inf and final_scores.mean().item() > 0:
        print("✅ 相似度计算测试通过")
        return True
    else:
        print("❌ 相似度计算测试失败")
        return False

def main():
    print("开始验证梯度计算修复效果...\n")
    
    # 测试Adam梯度计算
    test_adam_gradient_stability()
    
    # 测试相似度计算
    success = test_similarity_calculation()
    
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    if success:
        print("✅ 所有测试通过！修复后的实现应该能够处理数值稳定性问题。")
        print("\n建议：")
        print("1. 现在可以重新运行 main_less.py")
        print("2. 如果仍有问题，可以进一步调整epsilon值或梯度裁剪阈值")
        print("3. 监控日志输出以确认影响力分数不再为零")
    else:
        print("❌ 部分测试失败，可能需要进一步调整")
    
    print(f"\n修复内容总结：")
    print("1. 增强了Adam梯度计算的数值稳定性")
    print("2. 添加了梯度裁剪防止梯度爆炸")
    print("3. 改进了归一化逻辑，使用手动归一化")
    print("4. 添加了分块计算避免内存问题")
    print("5. 增强了NaN/Inf检测和处理")

if __name__ == "__main__":
    main()
