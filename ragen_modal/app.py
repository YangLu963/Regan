import modal

app = modal.App("ragen-simulated-webshop")

# 基础镜像配置
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "cmake")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.37.0", 
        "accelerate>=0.24.1",
        "numpy>=1.24.3",
        "requests>=2.31.0",
        "PyYAML>=6.0.1", 
        "urllib3>=2.0.0",
        "tqdm>=4.66.1",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "beautifulsoup4>=4.12.0"
    )  
    .run_commands(
        "git config --global http.postBuffer 1048576000"
    )
)

volume = modal.Volume.from_name("ragen-models", create_if_missing=True)

class SimulatedWebShopEnvironment:
    """模拟WebShop环境"""
    
    def __init__(self):
        self.products = self._generate_sample_products()
        self.current_state = None
        self.session_history = []
        
    def _generate_sample_products(self):
        """生成模拟产品数据"""
        products = []
        
        # 电子产品
        electronics = [
            {"id": "elec_001", "name": "iPhone 15 Pro", "category": "Electronics", "price": 999.99, "brand": "Apple", "attributes": {"storage": "128GB", "color": "Titanium", "screen": "6.1inch"}},
            {"id": "elec_002", "name": "Samsung Galaxy S24", "category": "Electronics", "price": 849.99, "brand": "Samsung", "attributes": {"storage": "256GB", "color": "Black", "screen": "6.2inch"}},
            {"id": "elec_003", "name": "MacBook Air M3", "category": "Electronics", "price": 1099.99, "brand": "Apple", "attributes": {"storage": "512GB", "color": "Space Gray", "screen": "13.6inch"}},
        ]
        
        # 服装
        clothing = [
            {"id": "cloth_001", "name": "Nike Air Max", "category": "Clothing", "price": 129.99, "brand": "Nike", "attributes": {"size": "10", "color": "White", "type": "Sneakers"}},
            {"id": "cloth_002", "name": "Adidas Hoodie", "category": "Clothing", "price": 59.99, "brand": "Adidas", "attributes": {"size": "M", "color": "Black", "type": "Hoodie"}},
        ]
        
        # 家居用品
        home = [
            {"id": "home_001", "name": "Stainless Steel Blender", "category": "Home", "price": 79.99, "brand": "KitchenAid", "attributes": {"capacity": "48oz", "color": "Silver", "power": "1000W"}},
            {"id": "home_002", "name": "Coffee Maker", "category": "Home", "price": 129.99, "brand": "Breville", "attributes": {"capacity": "12cup", "color": "Black", "type": "Drip"}},
        ]
        
        products.extend(electronics)
        products.extend(clothing)
        products.extend(home)
        return products
    
    def reset(self, user_query):
        """重置环境并设置用户查询"""
        self.current_state = {
            "query": user_query,
            "available_products": self.products.copy(),
            "filtered_products": self.products.copy(),
            "current_filters": {},
            "session_steps": 0,
            "completed": False,
            "reward": 0.0
        }
        self.session_history = [f"User query: {user_query}"]
        return self.current_state
    
    def apply_filter(self, filter_type, filter_value):
        """应用过滤器"""
        if self.current_state is None:
            return None
            
        self.current_state["current_filters"][filter_type] = filter_value
        self.current_state["filtered_products"] = [
            p for p in self.current_state["available_products"]
            if self._matches_filters(p, self.current_state["current_filters"])
        ]
        
        self.session_history.append(f"Applied filter: {filter_type} = {filter_value}")
        self.current_state["session_steps"] += 1
        
        return self.current_state
    
    def _matches_filters(self, product, filters):
        """检查产品是否匹配所有过滤器"""
        for filter_type, filter_value in filters.items():
            if filter_type in product.get("attributes", {}):
                if str(product["attributes"][filter_type]).lower() != str(filter_value).lower():
                    return False
            elif filter_type in product:
                if str(product[filter_type]).lower() != str(filter_value).lower():
                    return False
        return True
    
    def select_product(self, product_id):
        """选择产品"""
        if self.current_state is None:
            return None
            
        product = next((p for p in self.current_state["filtered_products"] if p["id"] == product_id), None)
        if product:
            self.current_state["completed"] = True
            self.current_state["selected_product"] = product
            self.current_state["reward"] = self._calculate_reward()
            self.session_history.append(f"Selected product: {product['name']}")
            
        return self.current_state
    
    def _calculate_reward(self):
        """计算奖励分数"""
        base_reward = 1.0
        efficiency_bonus = max(0, 1.0 - (self.current_state["session_steps"] * 0.1))
        return base_reward + efficiency_bonus
    
    def get_observation(self):
        """获取当前环境观察"""
        if self.current_state is None:
            return None
            
        return {
            "query": self.current_state["query"],
            "available_products_count": len(self.current_state["available_products"]),
            "filtered_products_count": len(self.current_state["filtered_products"]),
            "current_filters": self.current_state["current_filters"],
            "session_steps": self.current_state["session_steps"],
            "completed": self.current_state["completed"],
            "filtered_products": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "price": p["price"],
                    "brand": p["brand"],
                    "attributes": p["attributes"]
                }
                for p in self.current_state["filtered_products"][:5]  # 只返回前5个产品
            ]
        }

class SimulatedWebShopDataset:
    """模拟WebShop训练数据集"""
    
    def __init__(self):
        self.user_queries = [
            "I want to buy an iPhone with 128GB storage",
            "Looking for Nike sneakers in size 10",
            "Need a coffee maker that can make 12 cups",
            "I want a black Adidas hoodie in medium size",
            "Looking for a MacBook with 512GB storage",
            "Need a blender with at least 1000W power",
            "I want a Samsung phone with 256GB storage",
            "Looking for white Nike shoes",
            "Need a silver kitchen blender",
            "I want an Apple laptop in space gray color"
        ]
    
    def __len__(self):
        return len(self.user_queries)
    
    def __getitem__(self, idx):
        return self.user_queries[idx]
    
    def get_batch(self, batch_size=4):
        """获取批次数据"""
        import random
        batch_queries = random.sample(self.user_queries, min(batch_size, len(self.user_queries)))
        return batch_queries

class RAGENSimulatedTrainer:
    """在模拟环境中训练RAGEN"""
    
    def __init__(self):
        self.env = SimulatedWebShopEnvironment()
        self.dataset = SimulatedWebShopDataset()
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """初始化简单的策略模型"""
        # 这里可以替换为实际的LLM或强化学习模型
        print("🤖 初始化模拟训练模型...")
        return {"type": "simulated_policy", "initialized": True}
    
    def train_episode(self, user_query):
        """训练一个episode"""
        print(f"🎯 开始训练episode: {user_query}")
        
        # 重置环境
        state = self.env.reset(user_query)
        total_reward = 0
        steps = 0
        
        while not state["completed"] and steps < 10:  # 最多10步
            # 获取当前观察
            observation = self.env.get_observation()
            print(f"📊 Step {steps}: {len(observation['filtered_products'])} products available")
            
            # 模拟智能体动作（这里可以替换为实际的策略网络）
            action = self._simulate_agent_action(observation)
            
            # 执行动作
            if action["type"] == "filter":
                state = self.env.apply_filter(action["filter_type"], action["filter_value"])
                print(f"  → 应用过滤器: {action['filter_type']} = {action['filter_value']}")
            elif action["type"] == "select":
                state = self.env.select_product(action["product_id"])
                print(f"  → 选择产品: {action['product_id']}")
            
            steps += 1
        
        reward = state["reward"]
        total_reward += reward
        
        print(f"✅ Episode完成: 奖励={reward:.2f}, 步数={steps}")
        return total_reward
    
    def _simulate_agent_action(self, observation):
        """模拟智能体动作选择"""
        import random
        
        # 如果有过滤后的产品，随机选择一个
        if observation["filtered_products"] and random.random() < 0.7:
            product = random.choice(observation["filtered_products"])
            return {"type": "select", "product_id": product["id"]}
        
        # 否则应用随机过滤器
        available_filters = ["brand", "color", "storage", "size", "price_range"]
        filter_type = random.choice(available_filters)
        
        # 生成合理的过滤器值
        filter_values = {
            "brand": ["Apple", "Samsung", "Nike", "Adidas", "KitchenAid", "Breville"],
            "color": ["Black", "White", "Silver", "Space Gray", "Titanium"],
            "storage": ["128GB", "256GB", "512GB"],
            "size": ["M", "L", "10", "11"],
            "price_range": ["<100", "100-500", ">500"]
        }
        
        filter_value = random.choice(filter_values.get(filter_type, ["unknown"]))
        return {"type": "filter", "filter_type": filter_type, "filter_value": filter_value}
    
    def train(self, num_episodes=50):
        """主训练循环"""
        print("🚀 开始在模拟环境中训练RAGEN...")
        print(f"📈 计划训练 {num_episodes} 个episodes")
        
        total_rewards = []
        
        for episode in range(num_episodes):
            # 从数据集中获取用户查询
            user_query = self.dataset.get_batch(1)[0]
            
            # 训练一个episode
            reward = self.train_episode(user_query)
            total_rewards.append(reward)
            
            # 每10个episode打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = sum(total_rewards[-10:]) / 10
                print(f"📊 Episodes {episode-8}-{episode+1}: 平均奖励 = {avg_reward:.3f}")
        
        # 计算总体统计
        final_avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"🎉 训练完成! 最终平均奖励: {final_avg_reward:.3f}")
        
        # 保存训练结果
        self._save_training_results(total_rewards)
        
        return total_rewards
    
    def _save_training_results(self, rewards):
        """保存训练结果"""
        import json
        import numpy as np
        from pathlib import Path
        
        results = {
            "training_rewards": rewards,
            "average_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "total_episodes": len(rewards),
            "environment": "simulated_webshop"
        }
        
        # 保存到文件
        with open("training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print("💾 训练结果已保存到 training_results.json")

@app.function(
    image=base_image,
    gpu="A10G",
    timeout=3600,  # 1小时超时
    volumes={"/root/models": volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
def train_on_simulated_data():
    """在模拟数据上训练RAGEN"""
    import os
    import sys
    from pathlib import Path
    import subprocess
    import shutil
    
    print("🚀 开始模拟环境训练...")
    
    # 克隆GitHub仓库（可选，如果需要原始代码）
    repo_url = "https://github.com/YangLu963/Regan.git"
    work_dir = Path("/root/Regan")
    
    try:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        
        subprocess.run(
            ["git", "clone", repo_url, str(work_dir)],
            capture_output=True, text=True, check=True
        )
        print("✅ GitHub仓库克隆成功")
        
        # 切换到项目目录
        project_dir = work_dir / "ragen_modal"
        if project_dir.exists():
            os.chdir(project_dir)
            sys.path.insert(0, str(project_dir))
    except Exception as e:
        print(f"⚠️ GitHub克隆失败，使用本地模拟训练: {e}")
    
    # 开始模拟训练
    try:
        print("🎯 初始化模拟训练器...")
        trainer = RAGENSimulatedTrainer()
        
        print("🏋️ 开始训练循环...")
        rewards = trainer.train(num_episodes=50)
        
        # 保存结果到卷
        save_results_to_volume()
        
        return {
            "status": "completed",
            "message": "模拟训练成功完成",
            "average_reward": sum(rewards) / len(rewards),
            "total_episodes": len(rewards),
            "environment": "simulated"
        }
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def save_results_to_volume():
    """保存训练结果到共享卷"""
    import shutil
    from pathlib import Path
    import json
    
    print("\n💾 保存训练结果到卷...")
    saved_files = []
    
    # 创建模拟模型文件
    model_files = [
        "simulated_model_config.json",
        "training_results.json", 
        "training_log.txt"
    ]
    
    for filename in model_files:
        try:
            if filename == "simulated_model_config.json":
                config = {
                    "model_type": "RAGEN_Simulated",
                    "training_episodes": 50,
                    "environment": "SimulatedWebShop",
                    "version": "1.0"
                }
                with open(filename, "w") as f:
                    json.dump(config, f, indent=2)
            
            elif filename == "training_log.txt":
                with open(filename, "w") as f:
                    f.write("RAGEN Simulated Training Log\n")
                    f.write="Training completed successfully with simulated environment\n"
            
            # 复制到卷
            dest_path = Path("/root/models") / filename
            shutil.copy2(filename, dest_path)
            saved_files.append(filename)
            print(f"  ✅ 保存: {filename}")
            
        except Exception as e:
            print(f"  ⚠️ 保存 {filename} 失败: {e}")
    
    print(f"📦 总共保存了 {len(saved_files)} 个文件")

@app.function(
    image=base_image,
    volumes={"/root/models": volume}
)
def download_simulated_results():
    """下载模拟训练结果"""
    from pathlib import Path
    import shutil
    
    print("📥 下载模拟训练结果...")
    
    volume_path = Path("/root/models")
    local_path = Path(".")
    
    if not volume_path.exists():
        return {"status": "error", "message": "共享卷中没有数据"}
    
    downloaded_files = []
    for item in volume_path.iterdir():
        if item.is_file():
            shutil.copy2(item, local_path / item.name)
            downloaded_files.append(item.name)
            print(f"  ✅ 下载: {item.name}")
    
    return {"status": "success", "files": downloaded_files}

@app.function(image=base_image)
def test_simulated_environment():
    """测试模拟环境"""
    print("🧪 测试模拟WebShop环境...")
    
    env = SimulatedWebShopEnvironment()
    
    # 测试查询
    test_queries = [
        "I want to buy an iPhone with 128GB storage",
        "Looking for Nike sneakers in size 10"
    ]
    
    for query in test_queries:
        print(f"\n🔍 测试查询: '{query}'")
        state = env.reset(query)
        observation = env.get_observation()
        
        print(f"  可用产品: {observation['available_products_count']}")
        print(f"  过滤后产品: {observation['filtered_products_count']}")
        print(f"  当前过滤器: {observation['current_filters']}")
        
        # 显示前3个产品
        for i, product in enumerate(observation['filtered_products'][:3]):
            print(f"    {i+1}. {product['name']} - ${product['price']}")
    
    return {"status": "test_completed", "environment": "working"}

if __name__ == "__main__":
    with app.run():
        # 可以选择运行测试或训练
        test_simulated_environment.remote()
        train_on_simulated_data.remote()
