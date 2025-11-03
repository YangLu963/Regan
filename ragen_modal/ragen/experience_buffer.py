import numpy as np
import torch
from collections import deque
import random

class ExperienceBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, observation, instruction, think_content, action_content, reward, done, log_prob):
        """存储经验"""
        experience = {
            'observation': observation,
            'instruction': instruction,
            'think_content': think_content,
            'action_content': action_content,
            'reward': reward,
            'done': done,
            'log_prob': log_prob
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """采样一批经验"""
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        
        # 解包批次数据
        observations = [(exp['observation'], exp['instruction']) for exp in batch]
        think_contents = [exp['think_content'] for exp in batch]
        action_contents = [exp['action_content'] for exp in batch]
        rewards = [exp['reward'] for exp in batch]
        dones = [exp['done'] for exp in batch]
        log_probs = [exp['log_prob'] for exp in batch]
        
        return {
            'observations': observations,
            'think_contents': think_contents,
            'action_contents': action_contents,
            'rewards': torch.FloatTensor(rewards),
            'dones': torch.BoolTensor(dones),
            'log_probs': torch.FloatTensor(log_probs)
        }
    
    def clear(self):
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)