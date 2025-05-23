# -*- coding: utf-8 -*-
"""
Created on Sun May 11 13:56:28 2025

@author: 17808
"""

import gym
from jericho import FrotzEnv
from jericho.util import clean
from gym import spaces
import numpy as np
from config import Config

class ZorkEnv(gym.Env):
    def __init__(self, rom_path=Config.ROM_PATH):
        super(ZorkEnv, self).__init__()
        self.env = FrotzEnv(rom_path)
        
        # 定义动作和观察空间
        self.action_space = spaces.Discrete(20)  # 简化动作空间
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(1024,),  # 文本嵌入维度
            dtype=np.float32
        )
        
        # 预定义动作集
        self.actions = [
            'north', 'south', 'east', 'west', 
            'up', 'down', 'look', 'inventory',
            'take', 'drop', 'open', 'close',
            'eat', 'drink', 'read', 'turn on',
            'turn off', 'attack', 'ask', 'give'
        ]
        
    def reset(self):
        obs, info = self.env.reset()
        return self._process_obs(obs)
    
    def step(self, action):
        # 将离散动作转换为文本命令
        command = self.actions[action]
        obs, reward, done, info = self.env.step(command)
        return self._process_obs(obs), reward, done, info
    
    def _process_obs(self, obs):
        # 简化文本处理 - 实际项目中应使用更好的文本表示
        text = clean(obs)
        # 简单的词袋表示
        vec = np.zeros(1024)
        words = text.split()
        for word in words:
            vec[hash(word) % 1024] += 1
        return vec
    
    def close(self):
        self.env.close()