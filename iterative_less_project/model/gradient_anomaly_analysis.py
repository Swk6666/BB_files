#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
梯度异常分析：排查除精度外可能导致NaN/Inf的原因
"""

import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_potential_gradient_issues():
    """分析可能导致梯度异常的因素"""
    print("=" * 80)
    print("梯度异常原因分析（除精度问题外）")
    print("=" * 80)
    
    factors = [
        {
            "category": "1. 数值稳定性问题",
            "issues": [
                "除零操作：分母为0或接近0",
                "开方负数：sqrt(负数)",
                "对数域问题：log(0)或log(负数)",
                "指数爆炸：exp(过大值)",
                "梯度范数过大：梯度爆炸"
            ]
        },
        {
            "category": "2. 优化器状态异常",
            "issues": [
                "优化器状态不匹配：参数数量与状态不对应",
                "优化器状态损坏：exp_avg或exp_avg_sq包含异常值",
                "优化器状态精度问题：状态与梯度精度不兼容",
                "优化器初始化问题：状态未正确初始化",
                "优化器文件加载错误：文件损坏或格式不对"
            ]
        },
        {
            "category": "3. 模型相关问题", 
            "issues": [
                "模型权重异常：预训练模型包含NaN/Inf",
                "LoRA适配器问题：LoRA权重异常",
                "模型结构不匹配：实际结构与期望不符",
                "梯度检查点问题：gradient_checkpointing导致的问题",
                "注意力机制问题：attention score溢出"
            ]
        },
        {
            "category": "4. 数据相关问题",
            "issues": [
                "输入数据异常：input_ids包含无效token",
                "标签数据异常：labels包含异常值",
                "序列长度问题：过长序列导致内存/计算问题",
                "数据预处理错误：tokenization错误",
                "批次大小问题：过大批次导致数值不稳定"
            ]
        },
        {
            "category": "5. 硬件/环境问题",
            "issues": [
                "GPU内存不足：OOM导致计算异常",
                "CUDA版本不兼容：数值计算精度问题",
                "PyTorch版本问题：不同版本数值行为差异",
                "硬件故障：GPU硬件问题",
                "混合精度实现差异：不同框架的实现差异"
            ]
        },
        {
            "category": "6. Adam算法特定问题",
            "issues": [
                "Adam步数不匹配：step与状态不对应",
                "beta参数异常：beta1或beta2设置不当",
                "epsilon值过小：eps=1e-8可能在某些情况下不够",
                "动量累积异常：momentum过大导致不稳定",
                "学习率过高：与Adam状态结合导致爆炸"
            ]
        }
    ]
    
    for factor in factors:
        print(f"\n{factor['category']}")
        print("-" * 60)
        for i, issue in enumerate(factor['issues'], 1):
            print(f"  {i}. {issue}")

def create_diagnostic_tests():
    """创建诊断测试"""
    print("\n" + "=" * 80)
    print("诊断测试建议")
    print("=" * 80)
    
    tests = [
        {
            "test": "优化器状态检查",
            "commands": [
                "检查optimizer.pt文件完整性",
                "验证exp_avg和exp_avg_sq不包含NaN/Inf",
                "确认优化器状态参数数量与模型匹配",
                "打印优化器状态的数值范围"
            ]
        },
        {
            "test": "模型权重检查", 
            "commands": [
                "检查基础模型权重是否包含异常值",
                "验证LoRA适配器权重正常",
                "确认模型参数数量与预期一致",
                "检查模型的requires_grad设置"
            ]
        },
        {
            "test": "数据质量检查",
            "commands": [
                "验证训练数据的token有效性",
                "检查序列长度分布",
                "确认数据预处理正确性",
                "测试单个样本的前向传播"
            ]
        },
        {
            "test": "环境兼容性检查",
            "commands": [
                "确认PyTorch和CUDA版本兼容",
                "检查GPU内存使用情况",
                "验证混合精度设置",
                "测试简单的Adam优化器操作"
            ]
        }
    ]
    
    for test in tests:
        print(f"\n{test['test']}")
        print("-" * 40)
        for i, cmd in enumerate(test['commands'], 1):
            print(f"  {i}. {cmd}")

def specific_checks_for_your_case():
    """针对您的具体情况的检查建议"""
    print("\n" + "=" * 80)
    print("针对您的具体情况的检查建议")
    print("=" * 80)
    
    print("\n基于您的错误信息分析：")
    print("错误: '训练集梯度在归一化后出现NaN'")
    print("\n可能的原因（按优先级排序）:")
    
    causes = [
        {
            "priority": "高",
            "cause": "Adam优化器状态包含异常值",
            "check": "检查optimizer.pt中exp_avg_sq是否有负值或极大值",
            "solution": "重新训练warmup模型或使用不同的优化器设置"
        },
        {
            "priority": "高", 
            "cause": "优化器状态与模型参数不匹配",
            "check": "验证优化器状态参数数量与模型trainable parameters一致",
            "solution": "确保LoRA配置与训练时完全一致"
        },
        {
            "priority": "中",
            "cause": "LoRA适配器权重异常",
            "check": "检查warmup模型中的LoRA权重是否包含NaN/Inf",
            "solution": "重新训练warmup模型，使用更保守的学习率"
        },
        {
            "priority": "中",
            "cause": "数据中存在异常样本",
            "check": "逐个测试candidate_pool.jsonl中的样本",
            "solution": "过滤或修复异常数据"
        },
        {
            "priority": "低",
            "cause": "PyTorch/CUDA环境问题",
            "check": "在CPU上测试相同逻辑",
            "solution": "更新或重新安装PyTorch"
        }
    ]
    
    for cause in causes:
        print(f"\n[{cause['priority']}优先级] {cause['cause']}")
        print(f"  检查方法: {cause['check']}")
        print(f"  解决方案: {cause['solution']}")

if __name__ == "__main__":
    analyze_potential_gradient_issues()
    create_diagnostic_tests()
    specific_checks_for_your_case()
