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