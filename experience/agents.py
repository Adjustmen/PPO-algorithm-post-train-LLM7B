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
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessorList
)
from typing import List, Dict, Tuple
from collections import deque

class LLMAgent:
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        初始化LLM智能体
        Args:
            model_path: 预训练模型路径/名称
            device: 计算设备
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 历史记忆窗口
        self.memory = deque(maxlen=3)
        # 禁止动作列表（从失败中学习）
        self.forbidden_actions = set()
        
        # 生成配置默认值
        self.generation_config = GenerationConfig(
            max_new_tokens=40,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=5,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )

    def format_prompt(self, state: str) -> str:
        /*
        构建LLM输入提示模板
        Args:
            state: 当前游戏状态描述
        Returns:
            格式化后的提示文本
        */
        memory_str = "\n".join(self.memory)
        return f"""你正在玩Zork文本冒险游戏。请根据当前状态生成可能的动作。
        
游戏记忆（最近3步）：
{memory_str}

当前状态：
{state}

可参考的动作模板：
- 移动：go north/go south/go east/go west
- 交互：take [item]/use [item]/open [door]
- 观察：look/examine [object]

请生成5个最合理的动作，每行一个："""

    def generate_actions(self, state: str) -> List[str]:
        """
        生成候选动作列表
        Args:
            state: 当前游戏状态
        Returns:
            候选动作列表（已过滤无效动作）
        """
        prompt = self.format_prompt(state)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # 自定义logits处理器
        class ActionFilter(LogitsProcessor):
            def __call__(self, input_ids, scores):
                # 惩罚已知无效动作的token
                for token in self._get_forbidden_tokens():
                    scores[..., token] = -float('inf')
                return scores
        
        # 生成候选动作
        outputs = self.model.generate(
            **inputs,
            generation_config=self.generation_config,
            logits_processor=LogitsProcessorList([ActionFilter()])
        )
        
        # 解码并后处理
        raw_actions = self.tokenizer.batch_decode(
            outputs, 
            skip_special_tokens=True
        )
        return self._postprocess_actions(raw_actions)

    def _postprocess_actions(self, actions: List[str]) -> List[str]:
        """
        后处理生成的原始动作
        1. 提取动作部分（忽略生成的其他文本）
        2. 过滤语法错误动作
        3. 去重
        """
        processed = []
        for action in actions:
            # 提取第一个有效行
            lines = [l.strip() for l in action.split("\n") if l.strip()]
            if len(lines) > 1:  # 取生成的第一行动作
                action = lines[0]
            
            # 基础验证
            if self._validate_action(action):
                processed.append(action.lower())
        
        return list(set(processed))[:5]  # 去重后最多返回5个

    def _validate_action(self, action: str) -> bool:
        """验证动作基本语法"""
        if not action or len(action) > 50:
            return False
        if action in self.forbidden_actions:
            return False
        return any([
            action.startswith(verb) 
            for verb in ["go ", "take ", "use ", "open ", "examine "]
        ])

    def update_with_feedback(
        self,
        chosen_action: str,
        reward: float,
        next_state: str
    ):
        """
        根据RL反馈更新策略
        Args:
            chosen_action: 选择的动作
            reward: 获得的奖励
            next_state: 转移后的状态
        """
        # 记录到记忆
        self.memory.append(f"动作: {chosen_action} -> 奖励: {reward:.2f}")
        
        # 动态调整生成参数
        self._adapt_generation_config(reward)
        
        # 失败动作记录
        if reward < -1.0:
            self.forbidden_actions.add(chosen_action)

    def _adapt_generation_config(self, reward: float):
        """根据近期奖励调整生成参数"""
        # 探索阶段（奖励低时增加多样性）
        if reward < 0.5:
            self.generation_config.temperature = min(
                1.0, 
                self.generation_config.temperature + 0.05
            )
        # 利用阶段（奖励高时降低随机性）
        else:
            self.generation_config.temperature = max(
                0.3,
                self.generation_config.temperature - 0.02
            )

    def save_pretrained(self, output_dir: str):
        """保存模型和tokenizer"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        # 保存智能体特定状态
        torch.save({
            'forbidden_actions': list(self.forbidden_actions),
            'generation_config': self.generation_config
        }, f"{output_dir}/agent_state.pt")

    def load_pretrained(self, output_dir: str):
        """加载智能体状态"""
        state = torch.load(f"{output_dir}/agent_state.pt")
        self.forbidden_actions = set(state['forbidden_actions'])
        self.generation_config = state['generation_config']
