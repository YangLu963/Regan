import re

class RewardCalculator:
    def __init__(self):
        # 奖励权重配置
        self.weights = {
            'format_correct': 0.2,      # 格式正确
            'thinking_quality': 0.3,    # 思考质量
            'action_valid': 0.3,        # 动作有效
            'action_specific': 0.2,     # 动作具体性
            'task_relevant': 0.4,       # 任务相关
            'task_success': 1.0,        # 任务成功
            'step_efficiency': 0.1      # 步骤效率
        }
        
        # 任务关键词映射
        self.task_keywords = {
            'blanket': ['blanket', 'throw', 'quilt', 'cover', '毛毯', '毯子'],
            'jeans': ['jeans', 'denim', 'pants', 'trousers', '牛仔裤'],
            'laptop': ['laptop', 'computer', 'notebook', '笔记本电脑'],
            'shirt': ['shirt', 't-shirt', 'blouse', '衬衫'],
            'mouse': ['mouse', '无线鼠标', '鼠标'],
            'blue': ['blue', 'navy', 'azure', 'cobalt', '蓝色'],
            'red': ['red', 'crimson', 'scarlet', '红色'],
            'classic': ['classic', 'traditional', 'vintage', '经典'],
            'wireless': ['wireless', '蓝牙', '无线'],
            'size': ['size', '32', 'measurement', '尺寸']
        }
    
    def calculate_reward(self, think_content, action_content, env_feedback, task_success, instruction=None, step_number=None):
        """计算综合奖励 - 兼容新旧参数"""
        reward = 0.0
        reward_breakdown = {}
        
        print(f"\n🔍 奖励计算分析:")
        print(f"思考: {think_content}")
        print(f"动作: {action_content}")
        if instruction:
            print(f"任务: {instruction}")
        print(f"环境反馈: {env_feedback}")
        print(f"任务成功: {task_success}")
        
        # 1. 格式正确性奖励（改进版）
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
        if step_number is not None:
            efficiency_reward = self._calculate_efficiency_reward(step_number, task_success)
            reward += efficiency_reward
            reward_breakdown['efficiency'] = efficiency_reward
        
        # 显示奖励分解
        self._print_reward_breakdown(reward_breakdown, reward)
        
        return reward
    
    def calculate_simple_reward(self, think_content, action_content, task_success):
        """简化版奖励计算 - 确保向后兼容"""
        reward = 0.0
        
        print(f"\n🔍 简化奖励计算:")
        print(f"思考: {think_content}")
        print(f"动作: {action_content}")
        print(f"任务成功: {task_success}")
        
        # 基础格式奖励（不严格要求标签）
        has_think = think_content and len(think_content.strip()) > 5
        has_valid_action = action_content and self._is_valid_webshop_action(action_content)
        has_specific_action = action_content and action_content != "search[product]"
        
        if has_think:
            reward += 0.2
            print("✅ 思考奖励: +0.2")
        
        if has_valid_action:
            reward += 0.3
            print("✅ 动作格式奖励: +0.3")
            
            if has_specific_action:
                reward += 0.2
                print("✅ 具体动作奖励: +0.2")
        
        # 任务成功奖励
        if task_success:
            reward += 1.0
            print("🎉 任务成功奖励: +1.0")
        
        print(f"💎 总奖励: {reward:.2f}")
        return reward
    
    def _calculate_format_reward(self, think_content, action_content):
        """计算格式正确性奖励（改进版）"""
        format_score = 0.0
        
        # 检查思考内容是否有效（不严格要求标签）
        if think_content and len(think_content.strip()) > 10:
            if ("你的推理" not in think_content and 
                "请思考" not in think_content and 
                "思考过程" not in think_content):
                format_score += 0.1
                print("✅ 思考内容有效")
        
        # 检查动作格式
        if action_content:
            if self._is_valid_webshop_action(action_content):
                format_score += 0.1
                print("✅ 动作格式正确")
            else:
                print("❌ 动作格式错误")
        
        return format_score
    
    def _calculate_thinking_reward(self, think_content, instruction):
        """计算思考质量奖励"""
        if not think_content or len(think_content.strip()) < 15:
            print("❌ 思考内容过短")
            return 0.0
        
        thinking_score = 0.0
        
        # 检查是否包含任务分析
        if any(keyword in think_content.lower() for keyword in ['search', 'find', 'look', 'buy', 'purchase', 'click']):
            thinking_score += 0.1
            print("✅ 包含任务分析")
        
        # 检查是否包含推理过程
        if any(keyword in think_content.lower() for keyword in ['because', 'should', 'need', 'will', 'next', 'then']):
            thinking_score += 0.1
            print("✅ 包含推理过程")
        
        # 检查是否与环境相关
        if instruction and any(keyword in think_content.lower() for keyword in instruction.lower().split()):
            thinking_score += 0.1
            print("✅ 思考与任务相关")
        
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
            
            # 检查搜索关键词是否具体
            if len(action_content) > 12:  # search[product] 长度为13
                action_score += 0.05
                print("✅ 搜索关键词具体")
                
        elif action_content.startswith('click['):
            action_score += 0.2
            print("✅ 点击动作有效")
        elif action_content.startswith('buy['):
            action_score += 0.25
            print("✅ 购买动作有效")
        
        return action_score
    
    def _calculate_relevance_reward(self, think_content, action_content, instruction):
        """计算任务相关性奖励"""
        if not instruction:
            return 0.0
            
        relevance_score = 0.0
        instruction_lower = instruction.lower()
        
        # 根据任务类型检查相关性
        for product_type, keywords in self.task_keywords.items():
            if any(keyword in instruction_lower for keyword in keywords):
                # 检查思考相关性
                if any(keyword in think_content.lower() for keyword in keywords):
                    relevance_score += 0.1
                    print(f"✅ 思考与{product_type}相关")
                
                # 检查动作相关性
                if any(keyword in action_content.lower() for keyword in keywords):
                    relevance_score += 0.1
                    print(f"✅ 动作与{product_type}相关")
        
        return min(relevance_score, 0.4)  # 限制最大相关性奖励
    
    def _calculate_efficiency_reward(self, step_number, task_success):
        """计算步骤效率奖励"""
        if task_success:
            # 成功时，步骤越少奖励越高
            if step_number <= 3:
                return 0.1
            elif step_number <= 6:
                return 0.05
            elif step_number <= 10:
                return 0.02
        return 0.0
    
    def _is_valid_webshop_action(self, action):
        """检查动作格式有效性"""
        if not action:
            return False
        
        # 允许更灵活的动作格式
        valid_patterns = [
            r"^search\[.+\]$",
            r"^click\[\d+\]$", 
            r"^buy\[\d+\]$"
        ]
        
        action_clean = action.strip()
        return any(re.match(pattern, action_clean) for pattern in valid_patterns)
    
    def _print_reward_breakdown(self, breakdown, total_reward):
        """打印奖励分解详情"""
        print("\n📊 奖励分解:")
        for category, value in breakdown.items():
            if value > 0:
                print(f"  {category}: +{value:.2f}")
        print(f"💎 总奖励: {total_reward:.2f}")
        print("-" * 40)

# 测试函数
def test_reward_calculator():
    """测试奖励计算器"""
    print("🧪 测试奖励计算器...")
    
    calculator = RewardCalculator()
    
    # 测试案例
    test_cases = [
        {
            'think': '网页显示搜索页面，我需要搜索经典毛毯',
            'action': 'search[classic blanket]',
            'success': False,
            'instruction': 'Purchase a classic blanket'
        },
        {
            'think': '分析任务需求并搜索合适商品',
            'action': 'search[product]', 
            'success': False,
            'instruction': 'Get blue jeans'
        },
        {
            'think': '找到合适的笔记本电脑，价格低于1000美元',
            'action': 'buy[123]',
            'success': True,
            'instruction': 'Find a laptop under $1000'
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n📝 测试案例 {i+1}:")
        reward = calculator.calculate_reward(
            case['think'],
            case['action'], 
            "模拟环境反馈",
            case['success'],
            case['instruction'],
            1
        )
        print(f"最终奖励: {reward:.2f}")
        print("=" * 50)

if __name__ == "__main__":
    test_reward_calculator()
