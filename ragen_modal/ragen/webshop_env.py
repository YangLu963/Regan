# ragen/webshop_env.py - 修改后的版本
import requests
import json
import time
import random
import os

class WebShopEnv:
    def __init__(self, server_url="http://localhost:3000", max_steps=15):
        self.server_url = server_url
        self.max_steps = max_steps
        self.current_step = 0
        self.session_id = None
        
        # 关键修改：检查是否使用模拟模式
        self.use_simulation = os.environ.get("USE_SIMULATED_WEBSHOP", "true").lower() == "true"
        
        if self.use_simulation:
            print("🔧 使用WebShop模拟模式")
            # 初始化模拟数据
            self._init_simulation()
        else:
            print("🎯 使用真实WebShop环境")
            # 测试真实环境连接
            self._test_real_connection()
    
    def _init_simulation(self):
        """初始化模拟数据"""
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
        
        self.simulated_products = {
            'shirt': [{'id': 1, 'name': 'Red Cotton Shirt', 'color': 'red', 'price': 29.99}],
            'blanket': [{'id': 3, 'name': 'Classic Wool Blanket', 'type': 'classic', 'price': 49.99}],
            'jeans': [{'id': 5, 'name': 'Blue Denim Jeans Size 32', 'color': 'blue', 'size': 32, 'price': 59.99}],
            'laptop': [{'id': 7, 'name': 'Gaming Laptop $999', 'price': 999.99}],
            'mouse': [{'id': 9, 'name': 'Wireless Gaming Mouse', 'type': 'wireless', 'rating': 4.5, 'price': 49.99}]
        }
    
    def _test_real_connection(self):
        """测试真实WebShop连接"""
        try:
            response = requests.get(f"{self.server_url}/", timeout=5)
            if response.status_code == 200:
                print("✅ WebShop真实环境连接成功")
                return True
            else:
                print(f"❌ WebShop返回状态码 {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ WebShop真实环境连接失败: {e}")
            print("🔄 切换到模拟模式")
            self.use_simulation = True
            self._init_simulation()
            return False
    
    def reset(self, instruction=None):
        """重置环境"""
        self.current_step = 0
        
        if instruction is None:
            instruction = random.choice(self.tasks) if self.use_simulation else "Find a product"
        
        self.current_instruction = instruction
        
        if not self.use_simulation:
            try:
                # 真实环境
                response = requests.post(
                    f"{self.server_url}/reset", 
                    json={"instruction": instruction},
                    timeout=10
                )
                data = response.json()
                self.session_id = data.get('session_id', f'real_{int(time.time())}')
                observation = data.get('observation', f"真实环境: {instruction}")
                print(f"🎯 真实环境任务开始: {instruction}")
                return observation, {'session_id': self.session_id, 'instruction': instruction}
                
            except Exception as e:
                print(f"❌ 真实环境reset失败: {e}")
                print("🔄 切换到模拟模式")
                self.use_simulation = True
                self._init_simulation()
        
        # 模拟模式
        self.session_id = f"sim_{int(time.time())}"
        observation = f"欢迎！请{instruction}\n页面显示搜索框和商品分类。"
        
        print(f"🎯 模拟环境任务开始: {instruction}")
        return observation, {'session_id': self.session_id, 'instruction': instruction}
    
    def step(self, action, session_id=None):
        """执行动作"""
        if session_id is None:
            session_id = self.session_id
            
        self.current_step += 1
        
        if not self.use_simulation:
            try:
                # 真实环境
                payload = {'action': action, 'session_id': session_id}
                response = requests.post(f"{self.server_url}/step", json=payload, timeout=10)
                data = response.json()
                
                observation = data.get('observation', f"执行: {action}")
                reward = float(data.get('reward', 0.0))  # 修复：确保reward是float
                done = data.get('done', False) or self.current_step >= self.max_steps
                
                info = {
                    'session_id': session_id,
                    'step': self.current_step,
                    'action': action,
                    'real_environment': True
                }
                
                return observation, reward, done, info
                
            except Exception as e:
                print(f"❌ 真实环境step失败: {e}")
                self.use_simulation = True
        
        # 模拟模式
        observation, reward, done = self._simulate_step(action)
        
        info = {
            'session_id': session_id,
            'step': self.current_step,
            'action': action,
            'real_environment': False
        }
        
        return observation, float(reward), done, info  # 修复：确保reward是float
    
    def _simulate_step(self, action):
        """模拟环境步骤"""
        action_type = action.split('[')[0] if '[' in action else action
        
        if action_type == "search":
            reward = 0.2
            done = False
            observation = f"搜索结果页面 - 显示相关商品列表"
                
        elif action_type == "click":
            reward = 0.3
            done = False
            observation = f"商品详情页面 - 显示商品信息"
                
        elif action_type == "buy":
            success_prob = 0.6  # 基础成功率
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
            observation = "❌ 无效动作格式"
        
        # 步数限制
        if self.current_step >= self.max_steps and not done:
            done = True
            reward = 0.0
            observation = "⏰ 步数限制达到，任务失败"
        
        return observation, reward, done
    
    def close(self):
        """关闭环境"""
        if not self.use_simulation and self.session_id:
            try:
                requests.post(
                    f"{self.server_url}/close", 
                    json={'session_id': self.session_id},
                    timeout=5
                )
                print("✅ 真实环境关闭成功")
            except Exception as e:
                print(f"⚠️ 环境关闭失败: {e}")
