# -*- coding: utf-8 -*-
"""
Created on Wed May 14 17:04:47 2025

@author: 17808
"""
import torch
def generate_with_ppo_constraints(prompt, model, reward_model):
    # 1. 初始生成
    outputs = model.generate(
        prompt,
        output_scores=True,
        return_dict_in_generate=True,
        max_new_tokens=50
    )
    
    # 2. 计算各token的优势值
    advantages = []
    for i, token in enumerate(outputs.sequences[0]):
        # 使用奖励模型评估每个token
        reward = reward_model(prompt, token)
        baseline = reward_model.get_baseline(prompt)
        advantages.append(reward - baseline)
    
    # 3. 调整后续生成概率
    adjusted_probs = torch.softmax(
        outputs.scores[0] + 0.1 * torch.tensor(advantages),
        dim=-1
    )
    
    # 4. 从调整后的分布采样
    return torch.multinomial(adjusted_probs, num_samples=1)