import random
import numpy as np

class ExperienceBuffer:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, obs, instruction, think, action, reward, done, log_prob):
        experience = {
            'observation': obs,
            'instruction': instruction,
            'think': think,
            'action': action, 
            'reward': reward,
            'done': done,
            'log_prob': log_prob
        }
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
            
        batch = random.sample(self.buffer, batch_size)
        return {
            'observations': [(b['observation'], b['instruction']) for b in batch],
            'thinks': [b['think'] for b in batch],
            'actions': [b['action'] for b in batch],
            'rewards': [b['reward'] for b in batch],
            'dones': [b['done'] for b in batch],
            'log_probs': [b['log_prob'] for b in batch]
        }
        
    def __len__(self):
        return len(self.buffer)
