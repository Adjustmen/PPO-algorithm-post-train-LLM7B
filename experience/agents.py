# -*- coding: utf-8 -*-
"""
Created on Sun May 10 9:57:11 2025

@author: 17808
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class DQNAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        
        # 神经网络
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=config.LEARNING_RATE
        )
        self.steps_done = 0
    
    def _build_network(self):
        return nn.Sequential(
            nn.Linear(self.env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.env.action_space.n)
        )
    
    def act(self, observation):
        eps_threshold = self.config.EPS_END + (self.config.EPS_START - self.config.EPS_END) * \
            np.exp(-1. * self.steps_done / self.config.EPS_DECAY)
        self.steps_done += 1
        
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(torch.FloatTensor(observation)).argmax().item()
        else:
            return random.randrange(self.env.action_space.n)
    
    def update(self, obs, action, reward, next_obs, done):
        self.memory.append((obs, action, reward, next_obs, done))
        self._learn()
    
    def _learn(self):
        if len(self.memory) < self.config.BATCH_SIZE:
            return
        
        batch = random.sample(self.memory, self.config.BATCH_SIZE)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        
        obs = torch.FloatTensor(np.array(obs))
        next_obs = torch.FloatTensor(np.array(next_obs))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        current_q = self.policy_net(obs).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_obs).max(1)[0].detach()
        expected_q = rewards + (1 - dones) * self.config.GAMMA * next_q
        
        loss = nn.MSELoss()(current_q.squeeze(), expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())