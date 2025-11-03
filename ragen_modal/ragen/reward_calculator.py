import re

class RewardCalculator:
    def __init__(self):
        self.format_reward = 0.1
        self.thinking_reward = 0.2
        self.valid_action_reward = 0.3
        self.success_reward = 1.0
    
    def calculate_reward(self, think_content, action_content, env_feedback, task_success):
        """计算详细奖励（模仿教授的成功示例）"""
        reward = 0.0
        
        # 1. 格式正确性奖励（关键学习信号）
        if think_content and "<think>" in think_content and "</think>" in think_content:
            reward += self.format_reward
            print("✅ 格式正确奖励: +0.1")
        if action_content and "<action>" in action_content and "</action>" in action_content:
            reward += self.format_reward
            print("✅ 格式正确奖励: +0.1")
        
        # 2. 思考质量奖励
        if think_content and len(think_content) > 20:
            reward += self.thinking_reward
            print("✅ 思考质量奖励: +0.2")
        
        # 3. 动作有效性奖励
        if self._is_valid_webshop_action(action_content):
            reward += self.valid_action_reward
            print("✅ 有效动作奖励: +0.3")
        
        # 4. 任务成功奖励
        if task_success:
            reward += self.success_reward
            print("🎉 任务成功奖励: +1.0")
            
        print(f"总奖励: {reward:.2f}")
        return reward
    
    def _is_valid_webshop_action(self, action):
        """检查动作格式有效性"""
        if not action:
            return False
        valid_patterns = [
            r"search\[.*\]",
            r"click\[\d+\]", 
            r"buy\[\d+\]"
        ]
        return any(re.match(pattern, action.strip()) for pattern in valid_patterns)