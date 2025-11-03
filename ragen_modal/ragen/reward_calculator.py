import re

class RewardCalculator:
    def __init__(self):
        # 奖励权重配置
        self.weights = {
            'format_correct': 0.3,      # 格式正确
            'thinking_quality': 0.2,    # 思考质量
            'action_valid': 0.3,        # 动作有效
            'task_relevant': 0.4,       # 任务相关
            'task_success': 1.0,        # 任务成功
            'step_efficiency': 0.1      # 步骤效率
        }
        
        # 任务关键词映射
        self.task_keywords = {
            'blanket': ['blanket', 'throw', 'quilt', 'cover'],
            'jeans': ['jeans', 'denim', 'pants', 'trousers'],
            'blue': ['blue', 'navy', 'azure', 'cobalt'],
            'classic': ['classic', 'traditional', 'vintage', 'timeless'],
            'size': ['size', '32', 'measurement', 'waist']
        }
    
    def calculate_reward(self, think_content, action_content, env_feedback, task_success, instruction, step_number):
        """计算综合奖励"""
        reward = 0.0
        reward_breakdown = {}
        
        print(f"\n🔍 奖励计算分析:")
        print(f"思考: {think_content[:100]}...")
        print(f"动作: {action_content}")
        print(f"任务: {instruction}")
        
        # 1. 格式正确性奖励
        format_reward = self._calculate_format_reward(think_content, action_content)
        reward += format_reward
        reward_breakdown['format'] = format_reward
        
        # 2. 思考质量奖励
        thinking_reward = self._calculate_thinking_reward(think_content, instruction)
        reward += thinking_reward
        reward_breakdown['thinking'] = thinking_reward
        
        # 3. 动作有效性奖励
        action_reward = self._calculate_action_reward(action_content)
        reward += action_reward
        reward_breakdown['action'] = action_reward
        
        # 4. 任务相关性奖励
        relevance_reward = self._calculate_relevance_reward(think_content, action_content, instruction)
        reward += relevance_reward
        reward_breakdown['relevance'] = relevance_reward
        
        # 5. 任务成功奖励
        if task_success:
            success_reward = self.weights['task_success']
            reward += success_reward
            reward_breakdown['success'] = success_reward
            print("🎉 任务成功!")
        
        # 6. 步骤效率奖励（鼓励少步骤完成任务）
        efficiency_reward = self._calculate_efficiency_reward(step_number, task_success)
        reward += efficiency_reward
        reward_breakdown['efficiency'] = efficiency_reward
        
        # 显示奖励分解
        self._print_reward_breakdown(reward_breakdown, reward)
        
        return reward
    
    def _calculate_format_reward(self, think_content, action_content):
        """计算格式正确性奖励"""
        format_score = 0.0
        
        # 检查思考格式
        if think_content and len(think_content.strip()) > 10:
            if "思考" not in think_content and "你的推理" not in think_content:
                format_score += 0.15
                print("✅ 思考格式正确")
        
        # 检查动作格式
        if action_content:
            if re.match(r"^(search\[.*\]|click\[\d+\]|buy\[\d+\])$", action_content.strip()):
                format_score += 0.15
                print("✅ 动作格式正确")
            else:
                print("❌ 动作格式错误")
        
        return format_score
    
    def _calculate_thinking_reward(self, think_content, instruction):
        """计算思考质量奖励"""
        if not think_content or len(think_content.strip()) < 20:
            print("❌ 思考内容过短")
            return 0.0
        
        thinking_score = 0.0
        
        # 检查是否包含任务分析
        if any(keyword in think_content.lower() for keyword in ['search', 'find', 'look', 'buy']):
            thinking_score += 0.1
            print("✅ 包含任务分析")
        
        # 检查是否包含推理过程
        if any(keyword in think_content.lower() for keyword in ['because', 'should', 'need', 'will']):
            thinking_score += 0.1
            print("✅ 包含推理过程")
        
        return thinking_score
    
    def _calculate_action_reward(self, action_content):
        """计算动作有效性奖励"""
        if not action_content:
            print("❌ 无动作内容")
            return 0.0
        
        action_score = 0.0
        
        # 检查动作类型
        if action_content.startswith('search['):
            action_score += 0.15
            print("✅ 搜索动作有效")
        elif action_content.startswith('click['):
            action_score += 0.2
            print("✅ 点击动作有效")
        elif action_content.startswith('buy['):
            action_score += 0.25
            print("✅ 购买动作有效")
        
        # 检查动作内容是否合理
        if len(action_content) > 8:  # 基本的长度检查
            action_score += 0.05
            print("✅ 动作内容合理")
        
        return action_score
    
    def _calculate_relevance_reward(self, think_content, action_content, instruction):
        """计算任务相关性奖励"""
        relevance_score = 0.0
        instruction_lower = instruction.lower()
        
        # 根据任务类型检查相关性
        if 'blanket' in instruction_lower:
            if any(keyword in think_content.lower() for keyword in self.task_keywords['blanket']):
                relevance_score += 0.2
                print("✅ 思考与毯子任务相关")
            if any(keyword in action_content.lower() for keyword in self.task_keywords['blanket']):
                relevance_score += 0.2
                print("✅ 动作与毯子任务相关")
                
        elif 'jeans' in instruction_lower:
            if any(keyword in think_content.lower() for keyword in self.task_keywords['jeans']):
                relevance_score += 0.2
                print("✅ 思考与牛仔裤任务相关")
            if any(keyword in action_content.lower() for keyword in self.task_keywords['jeans']):
                relevance_score += 0.2
                print("✅ 动作与牛仔裤任务相关")
        
        # 检查颜色和尺寸要求
        if 'blue' in instruction_lower:
            if any(keyword in think_content.lower() for keyword in self.task_keywords['blue']):
                relevance_score += 0.1
            if any(keyword in action_content.lower() for keyword in self.task_keywords['blue']):
                relevance_score += 0.1
        
        if '32' in instruction_lower:
            if any(keyword in think_content.lower() for keyword in self.task_keywords['size']):
                relevance_score += 0.1
            if any(keyword in action_content.lower() for keyword in self.task_keywords['size']):
                relevance_score += 0.1
        
        return relevance_score
    
    def _calculate_efficiency_reward(self, step_number, task_success):
        """计算步骤效率奖励"""
        if task_success:
            # 成功时，步骤越少奖励越高
            if step_number <= 5:
                return 0.1
            elif step_number <= 10:
                return 0.05
        return 0.0
    
    def _print_reward_breakdown(self, breakdown, total_reward):
        """打印奖励分解详情"""
        print("\n📊 奖励分解:")
        for category, value in breakdown.items():
            print(f"  {category}: +{value:.2f}")
        print(f"💎 总奖励: {total_reward:.2f}")
        print("-" * 40)
    
    def calculate_simple_reward(self, think_content, action_content, task_success):
        """简化版奖励计算（用于测试）"""
        reward = 0.0
        
        # 基础格式奖励
        if think_content and len(think_content) > 10:
            reward += 0.2
        if action_content and any(x in action_content for x in ['search[', 'click[', 'buy[']):
            reward += 0.3
        
        # 任务成功奖励
        if task_success:
            reward += 1.0
        
        print(f"简化奖励: {reward:.2f}")
        return reward
