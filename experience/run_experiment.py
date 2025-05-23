# -*- coding: utf-8 -*-
"""
Created on Sun May 12 11:59:43 2025

@author: 17808
"""

import time
from tqdm import tqdm
from config import Config
from envs.zork_env import ZorkEnv
from agents.dqn_agent import DQNAgent
from utils.logger import Logger

def train():
    env = ZorkEnv()
    agent = DQNAgent(env, Config)
    logger = Logger()
    
    for episode in tqdm(range(Config.NUM_EPISODES), desc="Training"):
        obs = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            
            if done:
                break
        
        logger.log(episode, total_reward)
        
        if episode % Config.TARGET_UPDATE == 0:
            agent.update_target_network()
        
        if episode % Config.LOG_INTERVAL == 0:
            logger.print_stats(episode)
    
    env.close()
    logger.save("training_logs.csv")

if __name__ == "__main__":
    train()