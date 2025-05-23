# -*- coding: utf-8 -*-
"""
Created on Sun May 12 9:03:30 2025

@author: 17808
"""

def compute_reward(self, old_score, new_score, obs, done, valid_action):
    reward = new_score - old_score  # Game score change
    if done and "died" in obs.lower():  # Penalty for death
        reward -= 0.5
    if "new room" in obs:  # Placeholder for new location detection
        reward += 0.1
    if valid_action:  # Bonus for valid actions
        reward += 0.01
    # Add more conditions for object interaction, novelty, etc.
    return reward