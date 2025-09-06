#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断脚本：检查优化器状态和模型权重是否存在异常
"""

import torch
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from iterative_less import config

def check_optimizer_state():
    """检查优化器状态是否包含异常值"""
    print("=" * 80)
    print("检查优化器状态")
    print("=" * 80)
    
    warmup_model_path = os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_model")
    optimizer_path_pt = os.path.join(warmup_model_path, "optimizer.pt")
    optimizer_path_bin = os.path.join(warmup_model_path, "optimizer.bin")
    
    # 检查文件存在性
    if os.path.exists(optimizer_path_pt):
        optimizer_path = optimizer_path_pt
        print(f"✅ 找到优化器文件: {optimizer_path}")
    elif os.path.exists(optimizer_path_bin):
        optimizer_path = optimizer_path_bin
        print(f"✅ 找到优化器文件: {optimizer_path}")
    else:
        print(f"❌ 未找到优化器文件")
        print(f"   检查路径: {optimizer_path_pt}")
        print(f"   检查路径: {optimizer_path_bin}")
        return False
    
    try:
        # 加载优化器状态
        print("\n正在加载优化器状态...")
        optimizer_dict = torch.load(optimizer_path, map_location="cpu")
        print(f"✅ 优化器文件加载成功")
        
        # 检查结构
        print(f"\n优化器字典的键: {list(optimizer_dict.keys())}")
        
        if "state" in optimizer_dict:
            state = optimizer_dict["state"]
            print(f"✅ 找到 'state' 键，包含 {len(state)} 个参数状态")
        else:
            print(f"❌ 未找到 'state' 键")
            return False
        
        # 检查每个参数状态
        print(f"\n检查参数状态...")
        nan_count = 0
        inf_count = 0
        normal_count = 0
        
        for param_id, param_state in state.items():
            if 'exp_avg' in param_state and 'exp_avg_sq' in param_state:
                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']
                
                # 检查exp_avg
                if torch.isnan(exp_avg).any():
                    print(f"❌ 参数 {param_id} exp_avg 包含 NaN")
                    nan_count += 1
                elif torch.isinf(exp_avg).any():
                    print(f"❌ 参数 {param_id} exp_avg 包含 Inf") 
                    inf_count += 1
                    
                # 检查exp_avg_sq
                if torch.isnan(exp_avg_sq).any():
                    print(f"❌ 参数 {param_id} exp_avg_sq 包含 NaN")
                    nan_count += 1
                elif torch.isinf(exp_avg_sq).any():
                    print(f"❌ 参数 {param_id} exp_avg_sq 包含 Inf")
                    inf_count += 1
                elif (exp_avg_sq < 0).any():
                    print(f"❌ 参数 {param_id} exp_avg_sq 包含负值")
                    nan_count += 1
                else:
                    normal_count += 1
                    
                # 打印数值范围
                if param_id < 3:  # 只打印前3个参数的详细信息
                    print(f"参数 {param_id}:")
                    print(f"  exp_avg 范围: [{exp_avg.min().item():.2e}, {exp_avg.max().item():.2e}]")
                    print(f"  exp_avg_sq 范围: [{exp_avg_sq.min().item():.2e}, {exp_avg_sq.max().item():.2e}]")
                    print(f"  exp_avg_sq 最小值: {exp_avg_sq.min().item():.2e}")
        
        print(f"\n优化器状态检查结果:")
        print(f"  正常参数: {normal_count}")
        print(f"  包含NaN的参数: {nan_count}")
        print(f"  包含Inf的参数: {inf_count}")
        
        if nan_count > 0 or inf_count > 0:
            print(f"❌ 优化器状态包含异常值！这是梯度计算异常的主要原因。")
            return False
        else:
            print(f"✅ 优化器状态正常")
            return True
            
    except Exception as e:
        print(f"❌ 检查优化器状态时出错: {e}")
        return False

def check_model_weights():
    """检查模型权重是否正常"""
    print("\n" + "=" * 80)
    print("检查模型权重")
    print("=" * 80)
    
    warmup_model_path = os.path.join(config.ITERATIVE_OUTPUT_DIR, "warmup_model")
    
    try:
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        
        print(f"正在加载模型: {warmup_model_path}")
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME, 
            torch_dtype=torch.bfloat16,
            device_map="cpu"
        )
        
        # 加载LoRA适配器
        model = PeftModel.from_pretrained(base_model, warmup_model_path)
        
        print(f"✅ 模型加载成功")
        
        # 检查模型权重
        nan_params = []
        inf_params = []
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if torch.isnan(param).any():
                nan_params.append(name)
            elif torch.isinf(param).any():
                inf_params.append(name)
        
        print(f"\n模型权重检查结果:")
        print(f"  总参数数: {total_params}")
        print(f"  包含NaN的参数: {len(nan_params)}")
        print(f"  包含Inf的参数: {len(inf_params)}")
        
        if nan_params:
            print(f"❌ 包含NaN的参数:")
            for name in nan_params[:5]:  # 只显示前5个
                print(f"    - {name}")
                
        if inf_params:
            print(f"❌ 包含Inf的参数:")
            for name in inf_params[:5]:
                print(f"    - {name}")
        
        if len(nan_params) == 0 and len(inf_params) == 0:
            print(f"✅ 模型权重正常")
            return True
        else:
            print(f"❌ 模型权重包含异常值！")
            return False
            
    except Exception as e:
        print(f"❌ 检查模型权重时出错: {e}")
        return False

def check_data_samples():
    """检查数据样本是否包含异常"""
    print("\n" + "=" * 80)
    print("检查数据样本")
    print("=" * 80)
    
    try:
        import json
        
        # 检查候选池数据
        with open(config.CANDIDATE_POOL_FILENAME, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"候选池数据: {len(lines)} 行")
        
        # 检查前几个样本
        for i, line in enumerate(lines[:5]):
            try:
                data = json.loads(line)
                if 'text' in data:
                    text = data['text']
                    if len(text.strip()) == 0:
                        print(f"❌ 样本 {i} 文本为空")
                    elif len(text) > 10000:
                        print(f"⚠️  样本 {i} 文本过长: {len(text)} 字符")
                    else:
                        print(f"✅ 样本 {i} 正常: {len(text)} 字符")
                else:
                    print(f"❌ 样本 {i} 缺少 'text' 字段")
            except json.JSONDecodeError:
                print(f"❌ 样本 {i} JSON格式错误")
                
        return True
        
    except Exception as e:
        print(f"❌ 检查数据样本时出错: {e}")
        return False

def test_simple_gradient():
    """测试简单的梯度计算"""
    print("\n" + "=" * 80)
    print("测试简单梯度计算")
    print("=" * 80)
    
    try:
        # 创建简单测试
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=torch.float32)
        y = x.sum()
        y.backward()
        
        print(f"✅ 基础梯度计算正常")
        print(f"  x.grad: {x.grad}")
        
        # 测试Adam优化器
        from torch.optim import AdamW
        model = torch.nn.Linear(10, 1, dtype=torch.float32)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        
        x = torch.randn(5, 10, dtype=torch.float32)
        y = torch.randn(5, 1, dtype=torch.float32)
        
        for step in range(3):
            optimizer.zero_grad()
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            loss.backward()
            optimizer.step()
            
            # 检查梯度
            for name, param in model.named_parameters():
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"❌ 步骤 {step}: 参数 {name} 梯度异常")
                    return False
        
        print(f"✅ Adam优化器测试正常")
        return True
        
    except Exception as e:
        print(f"❌ 测试梯度计算时出错: {e}")
        return False

def main():
    """主诊断函数"""
    print("开始诊断梯度异常问题...")
    
    results = {
        "optimizer_state": check_optimizer_state(),
        "model_weights": check_model_weights(), 
        "data_samples": check_data_samples(),
        "simple_gradient": test_simple_gradient()
    }
    
    print("\n" + "=" * 80)
    print("诊断结果总结")
    print("=" * 80)
    
    for test, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test:20s}: {status}")
    
    failed_tests = [test for test, result in results.items() if not result]
    
    if not failed_tests:
        print(f"\n✅ 所有检查都通过！梯度异常可能是由精度问题引起的。")
    else:
        print(f"\n❌ 发现问题的检查项: {', '.join(failed_tests)}")
        print(f"\n建议优先解决这些问题，然后重新运行梯度计算。")

if __name__ == "__main__":
    main()
