# -*- coding: utf-8 -*-
"""
Created on Sun May 11 15:00:16 2025

@author: 17808
"""

from tqdm import tqdm
from config import Config
from envs.zork_env import ZorkEnv
from agents.dqn_agent import DQNAgent
import numpy as np

def evaluate(agent, num_episodes=10):
    env = ZorkEnv()
    scores = []
    steps = []
    
    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        obs = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        
        while not done:
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            if episode_steps >= Config.MAX_STEPS:
                break
        
        scores.append(episode_reward)
        steps.append(episode_steps)
    
    env.close()
    
    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "max_score": np.max(scores),
        "mean_steps": np.mean(steps),
        "completion_rate": np.mean([1 if s > 0 else 0 for s in scores])
    }

if __name__ == "__main__":
    env = ZorkEnv()
    agent = DQNAgent(env, Config)
    results = evaluate(agent)
    print("Evaluation Results:", results)