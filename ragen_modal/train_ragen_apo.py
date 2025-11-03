import torch
import torch.optim as optim
import numpy as np
import yaml
import os
from collections import deque
import time

from ragen.qwen_agent import QwenRAGENAgent
from ragen.experience_buffer import ExperienceBuffer
from ragen.webshop_env import WebShopEnv
from ragen.reward_calculator import RewardCalculator

class RAGENWebShopTrainer:
    def __init__(self, config_path="configs/webshop_config.yaml"):
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("=" * 60)
        print("RAGEN + A*PO + Qwen WebShop 训练系统")
        print("=" * 60)
        
        # 初始化组件
        self.env = WebShopEnv(
            server_url=self.config['environment']['server_url'],
            max_steps=self.config['environment']['max_steps']
        )
        
        self.agent = QwenRAGENAgent(
            model_name=self.config['model']['base_model'],
            device=self.config['model']['device']
        )
        
        self.reward_calculator = RewardCalculator()
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-5)  # 小学习率
        self.buffer = ExperienceBuffer(1000)
        
        # 训练统计
        self.episode_rewards = deque(maxlen=20)
        self.success_rates = deque(maxlen=20)
        self.format_success_rates = deque(maxlen=20)
        self.best_success_rate = 0.0
        self.total_steps = 0
        
    def collect_experience(self, num_episodes=2):
        """收集经验数据"""
        print(f"\n📥 收集 {num_episodes} 个回合的经验...")
        
        for episode in range(num_episodes):
            try:
                obs, info = self.env.reset()
                instruction = info['instruction']
                episode_reward = 0
                done = False
                steps = 0
                
                print(f"\n--- 回合 {episode+1} ---")
                print(f"任务: {instruction}")
                
                while not done and steps < self.config['environment']['max_steps']:
                    # 生成思考和动作
                    think_content, action_content, log_prob, full_response = self.agent.generate_webshop_response(obs, instruction)
                    
                    print(f"\n步骤 {steps+1}:")
                    print(f"思考: {think_content}")
                    print(f"动作: {action_content}")
                    
                    # 执行动作
                    next_obs, env_reward, done, info = self.env.step(action_content, info['session_id'])
                    
                    # 计算奖励
                    task_success = (env_reward > 0.5)
                    reward = self.reward_calculator.calculate_reward(think_content, action_content, next_obs, task_success)
                    
                    episode_reward += reward
                    steps += 1
                    self.total_steps += 1
                    
                    # 存储经验
                    self.buffer.push(obs, instruction, think_content, action_content, reward, done, log_prob)
                    
                    obs = next_obs
                    
                    if done:
                        break
                
                # 记录统计
                self.episode_rewards.append(episode_reward)
                success = 1 if episode_reward > 0.8 else 0
                self.success_rates.append(success)
                
                format_success = 1 if self._check_format_success(think_content, action_content) else 0
                self.format_success_rates.append(format_success)
                
                current_success = np.mean(self.success_rates) if self.success_rates else 0
                current_format = np.mean(self.format_success_rates) if self.format_success_rates else 0
                
                print(f"\n回合结果: 总奖励={episode_reward:.2f}, 成功率={current_success:.3f}, 格式成功率={current_format:.3f}")
                
            except Exception as e:
                print(f"回合 {episode+1} 出错: {e}")
                continue
    
    def _check_format_success(self, think_content, action_content):
        """检查格式是否正确"""
        has_think = think_content and len(think_content) > 5
        has_action = action_content and any(x in action_content for x in ['search[', 'click[', 'buy['])
        return has_think and has_action
    
    def train_step(self):
        """简化的训练步骤"""
        if len(self.buffer) < 4:  # 小批量
            return None
            
        batch = self.buffer.sample(4)
        if batch is None:
            return None
        
        # 计算优势（简化版）
        rewards = torch.FloatTensor(batch['rewards'])
        advantages = rewards - rewards.mean()
        
        # 策略损失
        log_probs = torch.FloatTensor(batch['log_probs'])
        policy_loss = -(log_probs * advantages).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': policy_loss.item(),
            'avg_reward': rewards.mean().item(),
            'avg_advantage': advantages.mean().item()
        }
    
    def train(self):
        """主训练循环"""
        print("\n🎯 开始训练...")
        print("成功标准: 成功率从0%提升到20%+")
        print("重点观察: Base Model学习格式遵循能力")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(50):  # 减少epoch数
            # 收集经验
            self.collect_experience(num_episodes=2)
            
            # 训练
            if len(self.buffer) >= 4:
                loss_info = self.train_step()
                
                if loss_info:
                    current_success = np.mean(self.success_rates) if self.success_rates else 0
                    current_format = np.mean(self.format_success_rates) if self.format_success_rates else 0
                    
                    print(f"Epoch {epoch:3d} | Loss: {loss_info['total_loss']:7.4f} | "
                          f"Reward: {loss_info['avg_reward']:5.3f} | "
                          f"Success: {current_success:5.3f} | Format: {current_format:5.3f}")
            
            # 评估
            if epoch % 5 == 0:
                current_success = np.mean(self.success_rates) if self.success_rates else 0
                current_format = np.mean(self.format_success_rates) if self.format_success_rates else 0
                
                if current_success > self.best_success_rate:
                    self.best_success_rate = current_success
                
                print(f"\n=== 评估 Epoch {epoch} ===")
                print(f"当前成功率: {current_success:.3f}")
                print(f"格式成功率: {current_format:.3f}")
                print(f"历史最佳: {self.best_success_rate:.3f}")
                
                # 成功标准检查
                if current_success >= 0.20:
                    print("🎉 达到Part 2作业要求: 成功率 > 20%!")
                    break
                    
                print("-" * 40)
        
        # 最终统计
        total_time = (time.time() - start_time) / 60
        final_success = np.mean(self.success_rates) if self.success_rates else 0
        
        print(f"\n" + "=" * 50)
        print("训练完成!")
        print(f"总训练时间: {total_time:.1f} 分钟")
        print(f"最终成功率: {final_success:.3f}")
        print(f"历史最佳成功率: {self.best_success_rate:.3f}")
        print("=" * 50)

def main():
    os.makedirs("configs", exist_ok=True)
    os.makedirs("ragen", exist_ok=True)
    
    trainer = RAGENWebShopTrainer()
    trainer.train()

if __name__ == "__main__":
    main()
