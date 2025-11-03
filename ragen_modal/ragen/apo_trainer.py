import torch
import torch.nn.functional as F
import pickle
import os
from collections import defaultdict
import numpy as np

class APOTrainer:
    def __init__(self, beta=0.02, gamma=0.99, cache_file="vstar_cache.pkl", num_vstar_samples=5):
        self.beta = beta
        self.gamma = gamma
        self.cache_file = cache_file
        self.num_vstar_samples = num_vstar_samples
        self.v_star_cache = self._load_cache()
        
    def _load_cache(self):
        """加载V*缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                    print(f"加载V*缓存，大小: {len(cache)}")
                    return defaultdict(float, cache)
            except Exception as e:
                print(f"缓存加载失败: {e}")
        return defaultdict(float)
    
    def _save_cache(self):
        """保存V*缓存"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(dict(self.v_star_cache), f)
        except Exception as e:
            print(f"缓存保存失败: {e}")
    
    def compute_advantages(self, batch_observations, batch_rewards, batch_dones, reference_agent, current_agent):
        """计算A*PO优势函数"""
        advantages = []
        v_star_values = []
        
        # 批量计算V*值（使用缓存）
        for i, (obs, instruction) in enumerate(batch_observations):
            # 使用观察和指令作为缓存键
            cache_key = f"{obs[:50]}_{instruction[:20]}"
            
            if cache_key not in self.v_star_cache:
                # 计算并缓存新的V*值
                self.v_star_cache[cache_key] = self._compute_v_star(obs, instruction, reference_agent)
            
            v_star_values.append(self.v_star_cache[cache_key])
        
        # 计算优势
        for i, (reward, done) in enumerate(zip(batch_rewards, batch_dones)):
            advantage = reward - v_star_values[i]
            advantages.append(advantage)
        
        # 定期保存缓存
        if len(batch_observations) % 10 == 0:
            self._save_cache()
            
        return torch.FloatTensor(advantages), torch.FloatTensor(v_star_values)
    
    def _compute_v_star(self, observation, instruction, reference_agent):
        """计算V*值（使用参考策略采样）"""
        total_value = 0.0
        
        # 用参考策略采样估算V*
        for _ in range(self.num_vstar_samples):
            _, _, _, full_response = reference_agent.generate_webshop_response(observation, instruction)
            # 使用响应质量作为价值估计（简化）
            value = self._estimate_response_value(full_response)
            total_value += value
        
        v_star = total_value / self.num_vstar_samples
        return v_star
    
    def _estimate_response_value(self, response):
        """估计响应的价值（基于格式正确性）"""
        value = 0.0
        if "<think>" in response and "</think>" in response:
            value += 0.3
        if "<action>" in response and "</action>" in response:
            value += 0.3
        if "search[" in response or "click[" in response or "buy[" in response:
            value += 0.4
        return value
    
    def compute_policy_loss(self, log_probs, advantages, ref_log_probs=None):
        """计算A*PO策略损失"""
        # 标准化优势函数
        advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 策略梯度部分
        policy_grad_loss = -torch.mean(advantages_normalized * log_probs)
        
        # KL散度惩罚
        kl_penalty = torch.tensor(0.0)
        if ref_log_probs is not None:
            kl_penalty = torch.mean(log_probs - ref_log_probs)
        
        total_loss = policy_grad_loss + self.beta * kl_penalty
        
        return total_loss, policy_grad_loss.item(), kl_penalty.item()