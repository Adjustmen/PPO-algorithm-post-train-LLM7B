# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:55:18 2025

@author: 17808
"""

import matplotlib.pyplot as plt
import numpy as np

# 模拟训练数据
episodes = np.arange(0, 1000, 10)
mean_rewards = np.clip(np.cumsum(np.random.normal(0.2, 0.1, 100)), 0, 20)
kl_divergence = np.abs(np.sin(episodes/100)) * 0.1 + 0.02
valid_actions = np.minimum(0.3 + 0.7 * (1 - np.exp(-episodes/200)), 0.95)

# 创建图表
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# 平均奖励曲线
ax1.plot(episodes, mean_rewards, color='#2ca02c', linewidth=2)
ax1.set_title('Average Reward per Episode', pad=10)
ax1.set_ylabel('Reward')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.axhline(y=15, color='r', linestyle='--', label='Human Baseline')
ax1.legend()

# KL散度曲线
ax2.plot(episodes, kl_divergence, color='#d62728', linewidth=2)
ax2.set_title('KL Divergence Control', pad=10)
ax2.set_ylabel('KL Value')
ax2.set_ylim(0, 0.15)
ax2.axhline(y=0.05, color='b', linestyle='--', label='Target KL')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

# 有效动作率曲线
ax3.plot(episodes, valid_actions*100, color='#1f77b4', linewidth=2)
ax3.set_title('Valid Action Rate', pad=10)
ax3.set_xlabel('Training Episodes')
ax3.set_ylabel('Percentage (%)')
ax3.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig('training_metrics.png', dpi=300)
plt.show()