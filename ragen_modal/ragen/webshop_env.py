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
        
        # 模拟任务库
        self.tasks = [
            "Find and buy a red shirt",
            "Purchase a classic blanket", 
            "Buy a wireless mouse with good ratings",
            "Find a laptop under $1000",
            "Get a blue jeans in size 32"
        ]
    
    def reset(self, instruction=None):
        """重置环境"""
        self.current_step = 0
        
        if instruction is None:
            instruction = random.choice(self.tasks)
        
        try:
            # 尝试连接真实WebShop环境
            response = requests.post(f"{self.server_url}/reset", 
                                   json={"instruction": instruction},
                                   timeout=5)
            data = response.json()
            self.session_id = data.get('session_id', 'simulated_session')
            observation = data.get('observation', f"模拟环境: {instruction}")
            
        except Exception as e:
            # 回退到模拟模式
            print(f"WebShop连接失败，使用模拟模式: {e}")
            self.session_id = f"simulated_{int(time.time())}"
            observation = f"欢迎！请{instruction}\n页面显示搜索框和商品列表。"
        
        print(f"🎯 任务开始: {instruction}")
        return observation, {'session_id': self.session_id, 'instruction': instruction}
    
    def step(self, action, session_id=None):
        """执行动作"""
        if session_id is None:
            session_id = self.session_id
            
        self.current_step += 1
        
        try:
            # 尝试真实环境
            payload = {'action': action, 'session_id': session_id}
            response = requests.post(f"{self.server_url}/step", json=payload, timeout=5)
            data = response.json()
            
            observation = data.get('observation', f"执行: {action}")
            reward = data.get('reward', 0.0)
            done = data.get('done', False) or self.current_step >= self.max_steps
            
        except Exception as e:
            # 模拟环境响应
            observation, reward, done = self._simulate_step(action)
        
        info = {
            'session_id': session_id,
            'step': self.current_step,
            'action': action
        }
        
        return observation, reward, done, info
    
    def _simulate_step(self, action):
        """模拟环境步骤"""
        # 基于动作给予奖励
        if "buy" in action and "1" in action:
            reward = 1.0
            done = True
            observation = "🎉 购买成功！任务完成。"
        elif "click" in action:
            reward = 0.3
            done = False
            observation = f"商品页面: 商品{action.split('[')[1].split(']')[0]}详情，可以购买。"
        elif "search" in action:
            reward = 0.1
            done = False
            query = action.split('[')[1].split(']')[0]
            observation = f"搜索结果: 找到5个{query}商品，请点击查看。"
        else:
            reward = -0.1
            done = False
            observation = "无效动作，请重试"
        
        # 步数限制
        if self.current_step >= self.max_steps:
            done = True
            reward = -0.5
            observation = "⏰ 步数限制，任务失败"
        
        return observation, reward, done
    
    def close(self):
        """关闭环境"""
        if self.session_id and 'simulated' not in self.session_id:
            try:
                requests.post(f"{self.server_url}/close", 
                            json={'session_id': self.session_id})
            except:
                pass