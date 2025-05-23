# -*- coding: utf-8 -*-
"""
Created on Sun May 11 14:00:52 2025

@author: 17808
"""

import csv
import time
from collections import deque

class Logger:
    def __init__(self, window_size=100):
        self.episode_rewards = []
        self.reward_window = deque(maxlen=window_size)
        self.start_time = time.time()
    
    def log(self, episode, reward):
        self.episode_rewards.append(reward)
        self.reward_window.append(reward)
    
    def print_stats(self, episode):
        mean_reward = sum(self.reward_window) / len(self.reward_window)
        elapsed = time.time() - self.start_time
        print(f"Episode {episode} - Mean Reward: {mean_reward:.2f} - Time: {elapsed:.2f}s")
    
    def save(self, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward"])
            for i, reward in enumerate(self.episode_rewards):
                writer.writerow([i, reward])