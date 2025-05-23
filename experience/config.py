# -*- coding: utf-8 -*-
"""
Created on Sun May 10 14:52:49 2025

@author: 17808
"""

class Config:
    # 游戏配置
    ROM_PATH = "/home/ubuntu/jericho/z-machine-games-master/jericho-game-suite/zork1.z5"  # 需要替换为实际ROM路径
    MAX_STEPS = 1000
    
    # DQN配置
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10
    LEARNING_RATE = 0.001
    MEMORY_SIZE = 10000
    
    # 训练配置
    NUM_EPISODES = 1000
    LOG_INTERVAL = 10
    EVAL_INTERVAL = 100