import requests
import json
import time
import random

class WebShopEnv:
    def __init__(self, server_url="http://localhost:3000", max_steps=15):
        self.server_url = server_url
        self.max_steps = max_steps
        self.current_step = 0
        self.session_id = None
        self.use_simulation = False
        self.current_instruction = None
        
        # 更丰富的模拟任务库
        self.tasks = [
            "Find and buy a red shirt",
            "Purchase a classic blanket", 
            "Buy a wireless mouse with good ratings",
            "Find a laptop under $1000",
            "Get a blue jeans in size 32",
            "Purchase a wireless keyboard",
            "Find a black backpack with laptop compartment",
            "Buy a stainless steel water bottle"
        ]
        
        # 模拟商品数据库
        self.simulated_products = {
            'shirt': [
                {'id': 1, 'name': 'Red Cotton Shirt', 'color': 'red', 'price': 29.99},
                {'id': 2, 'name': 'Blue Denim Shirt', 'color': 'blue', 'price': 39.99}
            ],
            'blanket': [
                {'id': 3, 'name': 'Classic Wool Blanket', 'type': 'classic', 'price': 49.99},
                {'id': 4, 'name': 'Modern Fleece Blanket', 'type': 'modern', 'price': 39.99}
            ],
            'jeans': [
                {'id': 5, 'name': 'Blue Denim Jeans Size 32', 'color': 'blue', 'size': 32, 'price': 59.99},
                {'id': 6, 'name': 'Black Skinny Jeans Size 32', 'color': 'black', 'size': 32, 'price': 49.99}
            ],
            'laptop': [
                {'id': 7, 'name': 'Gaming Laptop $999', 'price': 999.99},
                {'id': 8, 'name': 'Business Laptop $899', 'price': 899.99}
            ],
            'mouse': [
                {'id': 9, 'name': 'Wireless Gaming Mouse', 'type': 'wireless', 'rating': 4.5, 'price': 49.99},
                {'id': 10, 'name': 'Bluetooth Office Mouse', 'type': 'wireless', 'rating': 4.2, 'price': 29.99}
            ]
        }
        
        # 测试连接
        self._test_connection()
    
    def _test_connection(self):
        """测试WebShop连接"""
        try:
            response = requests.get(f"{self.server_url}/", timeout=3)
            if response.status_code == 200:
                print("✅ WebShop连接成功")
                return True
        except Exception as e:
            print(f"⚠️ WebShop连接失败，使用模拟模式: {e}")
            self.use_simulation = True
            return False
    
    def reset(self, instruction=None):
        """重置环境"""
        self.current_step = 0
        
        if instruction is None:
            instruction = random.choice(self.tasks)
        
        self.current_instruction = instruction
        
        if not self.use_simulation:
            try:
                # 尝试连接真实WebShop环境
                response = requests.post(
                    f"{self.server_url}/reset", 
                    json={"instruction": instruction},
                    timeout=5
                )
                data = response.json()
                self.session_id = data.get('session_id', f'real_{int(time.time())}')
                observation = data.get('observation', f"真实环境: {instruction}")
                print(f"🎯 任务开始: {instruction}")
                return observation, {'session_id': self.session_id, 'instruction': instruction}
                
            except Exception as e:
                print(f"❌ 真实环境失败，切换到模拟模式: {e}")
                self.use_simulation = True
        
        # 模拟模式
        self.session_id = f"sim_{int(time.time())}"
        observation = self._get_simulated_observation("reset", instruction)
        
        print(f"🎯 任务开始 (模拟模式): {instruction}")
        return observation, {'session_id': self.session_id, 'instruction': instruction}
    
    def step(self, action, session_id=None):
        """执行动作"""
        if session_id is None:
            session_id = self.session_id
            
        self.current_step += 1
        
        if not self.use_simulation:
            try:
                # 尝试真实环境
                payload = {'action': action, 'session_id': session_id}
                response = requests.post(f"{self.server_url}/step", json=payload, timeout=5)
                data = response.json()
                
                observation = data.get('observation', f"执行: {action}")
                reward = data.get('reward', 0.0)
                done = data.get('done', False) or self.current_step >= self.max_steps
                
                info = {
                    'session_id': session_id,
                    'step': self.current_step,
                    'action': action
                }
                
                return observation, reward, done, info
                
            except Exception as e:
                print(f"❌ 真实环境步骤失败: {e}")
                self.use_simulation = True
        
        # 模拟模式
        observation, reward, done = self._simulate_step(action)
        
        info = {
            'session_id': session_id,
            'step': self.current_step,
            'action': action,
            'simulated': True
        }
        
        return observation, reward, done, info
    
    def _get_simulated_observation(self, state, instruction):
        """获取模拟观察"""
        observations = {
            "reset": f"欢迎！请{instruction}\n页面显示搜索框和商品分类。",
            "search": f"搜索结果页面 - 显示相关商品列表。任务: {instruction}",
            "product": "商品详情页面 - 显示商品信息、价格和评价。",
            "cart": "购物车页面 - 显示已选商品和总价。",
            "checkout": "结算页面 - 确认订单信息。"
        }
        return observations.get(state, f"当前状态: {state}")
    
    def _simulate_step(self, action):
        """改进的模拟环境步骤"""
        # 解析动作
        action_type = action.split('[')[0] if '[' in action else action
        action_content = action.split('[')[1].split(']')[0] if '[' in action else ""
        
        # 基于动作类型和内容给予奖励
        if action_type == "search":
            reward = 0.2
            done = False
            # 检查搜索关键词是否相关
            if self._is_relevant_search(action_content, self.current_instruction):
                reward += 0.1
                observation = f"✅ 相关搜索结果: 找到多个匹配'{action_content}'的商品"
            else:
                observation = f"❌ 搜索结果: 未找到高度相关的'{action_content}'商品"
                
        elif action_type == "click":
            reward = 0.3
            done = False
            try:
                product_id = int(action_content)
                observation = f"📦 商品{product_id}详情: 可查看详情并购买"
            except:
                observation = f"📦 商品详情页面"
                
        elif action_type == "buy":
            # 智能成功率计算
            success_prob = self._calculate_success_probability(action, self.current_instruction)
            if random.random() < success_prob:
                reward = 1.0
                done = True
                observation = "🎉 购买成功！任务完成！"
            else:
                reward = 0.1
                done = False
                observation = "⚠️ 购买失败，请检查商品或重试"
                
        else:
            reward = -0.1
            done = False
            observation = "❌ 无效动作格式，请使用: search[关键词], click[ID], buy[ID]"
        
        # 步数限制
        if self.current_step >= self.max_steps and not done:
            done = True
            reward = 0.0  # 改为0而不是负奖励
            observation = "⏰ 步数限制达到，任务失败"
        
        return observation, reward, done
    
    def _is_relevant_search(self, search_query, instruction):
        """检查搜索关键词是否与任务相关"""
        instruction_lower = instruction.lower()
        query_lower = search_query.lower()
        
        # 关键词匹配
        if "red shirt" in instruction_lower and "red" in query_lower and "shirt" in query_lower:
            return True
        elif "classic blanket" in instruction_lower and "classic" in query_lower and "blanket" in query_lower:
            return True
        elif "blue jeans" in instruction_lower and "blue" in query_lower and "jeans" in query_lower:
            return True
        elif "laptop" in instruction_lower and "laptop" in query_lower:
            return True
        elif "wireless mouse" in instruction_lower and "wireless" in query_lower and "mouse" in query_lower:
            return True
            
        return False
    
    def _calculate_success_probability(self, action, instruction):
        """计算购买成功概率"""
        base_prob = 0.3
        
        # 基于任务相关性的加成
        if self._is_relevant_search(action, instruction):
            base_prob += 0.3
            
        # 基于步骤效率的加成（越早购买成功率越高）
        if self.current_step <= 5:
            base_prob += 0.2
        elif self.current_step <= 10:
            base_prob += 0.1
            
        return min(base_prob, 0.8)  # 最大80%成功率
    
    def close(self):
        """关闭环境"""
        if not self.use_simulation and self.session_id and 'sim_' not in self.session_id:
            try:
                requests.post(
                    f"{self.server_url}/close", 
                    json={'session_id': self.session_id},
                    timeout=3
                )
                print("✅ 环境关闭成功")
            except Exception as e:
                print(f"⚠️ 环境关闭失败: {e}")