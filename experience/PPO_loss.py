# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:05:52 2025

@author: 17808
"""
import torch

def ppo_loss(new_logprobs, old_logprobs, advantages, clip_epsilon=0.2):
    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
    
    # 添加KL散度约束
    kl_penalty = 0.01 * (new_logprobs.exp() * 
                        (new_logprobs - old_logprobs)).sum()
    
    return -torch.min(ratio * advantages, clipped_ratio * advantages).mean() + kl_penalty
# src/rl/algorithms/ppo.py
class AdvancedPPOTrainer:
    def __init__(self, ...):
        # 动态调整的超参数
        self.kl_coef = 0.2       # KL惩罚系数（初始值）
        self.clip_range = 0.2    # 策略裁剪范围
        self.gamma = 0.99        # 折扣因子
        self.lam = 0.95          # GAE系数
        self.entropy_coef = 0.01 # 熵奖励系数
        
    def update(self, samples):
        # 动态调整KL系数
        if self.kl_mean > 0.03:  # 监控KL散度
            self.kl_coef *= 1.5
        elif self.kl_mean < 0.01:
            self.kl_coef *= 0.8
        
        # 自适应clip_range
        self.clip_range = max(0.1, 0.2 * (1 - self.update_step/1e5))
