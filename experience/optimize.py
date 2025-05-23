# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:35:09 2025

@author: 17808
"""

# 梯度检查点
model.gradient_checkpointing_enable()

# 8-bit Adam优化器
import bitsandbytes as bnb
optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=1e-5,
    betas=(0.9, 0.999)
def dynamic_batching(observations, max_len=256):
    # 按长度排序
    sorted_obs = sorted(observations, key=lambda x: len(x), reverse=True)
    batches = []
    current_batch = []
    current_len = 0
    
    for obs in sorted_obs:
        obs_len = len(obs)
        if current_len + obs_len > max_len:
            batches.append(current_batch)
            current_batch = []
            current_len = 0
        current_batch.append(obs)
        current_len += obs_len
    
    if current_batch:
        batches.append(current_batch)
    return batches