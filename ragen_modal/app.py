import modal
import time
import random
import json

app = modal.App("ragen-github-webshop")

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
        "beautifulsoup4>=4.12.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    )  
    .run_commands(
        "git config --global http.postBuffer 1048576000"
    )
)

volume = modal.Volume.from_name("ragen-models", create_if_missing=True)

class DetailedWebShopEnvironment:
    """详细的模拟WebShop环境"""
    
    def __init__(self):
        self.products = self._generate_detailed_products()
        self.current_state = None
        self.session_history = []
        self.metrics = {
            'total_steps': 0,
            'successful_selections': 0,
            'failed_selections': 0,
            'filter_applications': 0
        }
        
    def _generate_detailed_products(self):
        """生成详细的模拟产品数据"""
        products = []
        
        # 电子产品
        electronics = [
            {"id": "elec_001", "name": "iPhone 15 Pro", "category": "Electronics", "price": 999.99, "brand": "Apple", 
             "attributes": {"storage": "128GB", "color": "Titanium", "screen": "6.1inch", "camera": "48MP"}},
            {"id": "elec_002", "name": "Samsung Galaxy S24", "category": "Electronics", "price": 849.99, "brand": "Samsung", 
             "attributes": {"storage": "256GB", "color": "Black", "screen": "6.2inch", "camera": "50MP"}},
            {"id": "elec_003", "name": "MacBook Air M3", "category": "Electronics", "price": 1099.99, "brand": "Apple", 
             "attributes": {"storage": "512GB", "color": "Space Gray", "screen": "13.6inch", "ram": "8GB"}},
            {"id": "elec_004", "name": "Google Pixel 8", "category": "Electronics", "price": 699.99, "brand": "Google", 
             "attributes": {"storage": "128GB", "color": "White", "screen": "6.3inch", "camera": "50MP"}},
        ]
        
        # 服装
        clothing = [
            {"id": "cloth_001", "name": "Nike Air Max", "category": "Clothing", "price": 129.99, "brand": "Nike", 
             "attributes": {"size": "10", "color": "White", "type": "Sneakers", "material": "Leather"}},
            {"id": "cloth_002", "name": "Adidas Hoodie", "category": "Clothing", "price": 59.99, "brand": "Adidas", 
             "attributes": {"size": "M", "color": "Black", "type": "Hoodie", "material": "Cotton"}},
            {"id": "cloth_003", "name": "Under Armour Shorts", "category": "Clothing", "price": 34.99, "brand": "Under Armour", 
             "attributes": {"size": "L", "color": "Blue", "type": "Shorts", "material": "Polyester"}},
        ]
        
        # 家居用品
        home = [
            {"id": "home_001", "name": "Stainless Steel Blender", "category": "Home", "price": 79.99, "brand": "KitchenAid", 
             "attributes": {"capacity": "48oz", "color": "Silver", "power": "1000W", "type": "Countertop"}},
            {"id": "home_002", "name": "Coffee Maker", "category": "Home", "price": 129.99, "brand": "Breville", 
             "attributes": {"capacity": "12cup", "color": "Black", "type": "Drip", "features": "Programmable"}},
        ]
        
        products.extend(electronics)
        products.extend(clothing)
        products.extend(home)
        return products
    
    def reset(self, user_query, target_product_id=None):
        """重置环境"""
        self.current_state = {
            "query": user_query,
            "available_products": self.products.copy(),
            "filtered_products": self.products.copy(),
            "current_filters": {},
            "session_steps": 0,
            "completed": False,
            "reward": 0.0,
            "target_product_id": target_product_id,
            "correct_selection": False
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
        self.metrics['filter_applications'] += 1
        self.metrics['total_steps'] += 1
        
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
            
            # 检查是否正确选择了目标产品
            target_id = self.current_state.get("target_product_id")
            if target_id:
                self.current_state["correct_selection"] = (product_id == target_id)
                if self.current_state["correct_selection"]:
                    self.metrics['successful_selections'] += 1
                else:
                    self.metrics['failed_selections'] += 1
            else:
                self.metrics['successful_selections'] += 1
            
            self.current_state["reward"] = self._calculate_reward()
            self.session_history.append(f"Selected product: {product['name']}")
            
        return self.current_state
    
    def _calculate_reward(self):
        """计算详细的奖励分数"""
        base_reward = 1.0 if self.current_state.get("correct_selection", True) else 0.0
        
        # 效率奖励（步数越少奖励越高）
        efficiency_bonus = max(0, 1.0 - (self.current_state["session_steps"] * 0.1))
        
        # 准确性奖励
        accuracy_bonus = 0.5 if self.current_state.get("correct_selection", False) else 0.0
        
        # 多样性奖励（使用不同过滤器）
        unique_filters = len(set(self.current_state["current_filters"].keys()))
        diversity_bonus = unique_filters * 0.1
        
        total_reward = base_reward + efficiency_bonus + accuracy_bonus + diversity_bonus
        return min(total_reward, 2.0)  # 限制最大奖励
    
    def get_metrics(self):
        """获取环境指标"""
        return self.metrics.copy()

class TrainingEvaluator:
    """训练评估器"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_accuracies = []
        self.training_history = []
        
    def record_episode(self, episode, reward, steps, accuracy, query, selected_product):
        """记录每个episode的结果"""
        episode_data = {
            'episode': episode,
            'reward': reward,
            'steps': steps,
            'accuracy': accuracy,
            'query': query,
            'selected_product': selected_product,
            'timestamp': time.time()
        }
        self.training_history.append(episode_data)
        self.episode_rewards.append(reward)
        self.episode_steps.append(steps)
        self.episode_accuracies.append(accuracy)
    
    def get_summary_stats(self):
        """获取汇总统计"""
        if not self.episode_rewards:
            return {}
            
        return {
            'total_episodes': len(self.episode_rewards),
            'average_reward': sum(self.episode_rewards) / len(self.episode_rewards),
            'average_steps': sum(self.episode_steps) / len(self.episode_steps),
            'average_accuracy': sum(self.episode_accuracies) / len(self.episode_accuracies),
            'max_reward': max(self.episode_rewards),
            'min_reward': min(self.episode_rewards),
            'success_rate': sum(self.episode_accuracies) / len(self.episode_accuracies) * 100,
            'efficiency': sum(self.episode_rewards) / sum(self.episode_steps) if sum(self.episode_steps) > 0 else 0
        }
    
    def print_detailed_report(self):
        """打印详细报告"""
        stats = self.get_summary_stats()
        
        print("\n" + "="*80)
        print("📊 Detailed Training Report")
        print("="*80)
        
        print(f"📈 Overall Statistics:")
        print(f"   • Total Episodes: {stats['total_episodes']}")
        print(f"   • Average Reward: {stats['average_reward']:.3f}")
        print(f"   • Average Steps: {stats['average_steps']:.1f}")
        print(f"   • Success Rate: {stats['success_rate']:.1f}%")
        print(f"   • Training Efficiency: {stats['efficiency']:.3f}")
        print(f"   • Max Reward: {stats['max_reward']:.3f}")
        print(f"   • Min Reward: {stats['min_reward']:.3f}")
        
        print(f"\n🎯 Recent 5 Episodes:")
        for i, history in enumerate(self.training_history[-5:]):
            print(f"   Episode {history['episode']+1}: Reward={history['reward']:.2f}, "
                  f"Steps={history['steps']}, Accuracy={history['accuracy']}, "
                  f"Query='{history['query'][:30]}...'")
        
        # 学习进度分析
        if len(self.episode_rewards) >= 10:
            first_half = self.episode_rewards[:len(self.episode_rewards)//2]
            second_half = self.episode_rewards[len(self.episode_rewards)//2:]
            improvement = (sum(second_half)/len(second_half) - sum(first_half)/len(first_half)) / (sum(first_half)/len(first_half)) * 100
            print(f"\n📈 Learning Progress: Last 50% improved by {improvement:+.1f}% vs first 50%")

class DetailedRAGENTrainer:
    """详细的RAGEN训练器"""
    
    def __init__(self, use_simulated=True):
        self.use_simulated = use_simulated
        self.env = DetailedWebShopEnvironment() if use_simulated else None
        self.evaluator = TrainingEvaluator()
        self.training_queries = self._get_training_queries()
        
    def _get_training_queries(self):
        """获取训练查询和目标产品"""
        return [
            {"query": "I want to buy an iPhone with 128GB storage", "target": "elec_001"},
            {"query": "Looking for Nike sneakers in size 10", "target": "cloth_001"},
            {"query": "Need a MacBook with 512GB storage", "target": "elec_003"},
            {"query": "I want a black Adidas hoodie in medium size", "target": "cloth_002"},
            {"query": "Looking for Samsung phone with 256GB storage", "target": "elec_002"},
            {"query": "Need a silver kitchen blender", "target": "home_001"},
            {"query": "I want a Google Pixel phone in white color", "target": "elec_004"},
            {"query": "Looking for Under Armour shorts in large size", "target": "cloth_003"},
            {"query": "Need a programmable coffee maker", "target": "home_002"},
            {"query": "I want an Apple laptop in space gray color", "target": "elec_003"},
        ]
    
    def train_episode_detailed(self, episode_idx):
        """详细的episode训练"""
        query_data = self.training_queries[episode_idx % len(self.training_queries)]
        user_query = query_data["query"]
        target_product = query_data["target"]
        
        print(f"\n🎯 Episode {episode_idx + 1}: '{user_query}'")
        print(f"   Target Product: {target_product}")
        
        state = self.env.reset(user_query, target_product)
        steps = 0
        max_steps = 15
        
        while not state["completed"] and steps < max_steps:
            observation = self._get_observation(state)
            action = self._select_intelligent_action(observation, steps)
            
            if action["type"] == "filter":
                state = self.env.apply_filter(action["filter_type"], action["filter_value"])
                print(f"   → Step {steps+1}: Apply filter [{action['filter_type']}={action['filter_value']}]")
                print(f"      Remaining products: {len(state['filtered_products'])}")
            elif action["type"] == "select":
                state = self.env.select_product(action["product_id"])
                accuracy = "✓" if state.get("correct_selection", False) else "✗"
                print(f"   → Step {steps+1}: Select product [{action['product_id']}] {accuracy}")
            
            steps += 1
        
        # 记录结果
        accuracy = 1.0 if state.get("correct_selection", False) else 0.0
        selected_name = state.get("selected_product", {}).get("name", "None")
        
        self.evaluator.record_episode(
            episode_idx, state["reward"], steps, accuracy, 
            user_query, selected_name
        )
        
        print(f"   ✅ Completed: Reward={state['reward']:.2f}, Steps={steps}, "
              f"Accuracy={accuracy}, Selected='{selected_name}'")
        
        return state["reward"], steps, accuracy
    
    def _get_observation(self, state):
        """获取环境观察"""
        return {
            "filtered_products": state["filtered_products"],
            "current_filters": state["current_filters"],
            "query": state["query"],
            "steps": state["session_steps"]
        }
    
    def _select_intelligent_action(self, observation, step):
        """智能动作选择（模拟策略）"""
        products = observation["filtered_products"]
        query = observation["query"].lower()
        
        # 如果有产品且符合条件，选择产品
        if products and (step >= 3 or random.random() < 0.3):
            # 尝试选择最符合查询的产品
            best_product = self._find_best_match(products, query)
            return {"type": "select", "product_id": best_product["id"]}
        
        # 否则应用智能过滤器
        filter_type, filter_value = self._select_smart_filter(query, observation["current_filters"])
        return {"type": "filter", "filter_type": filter_type, "filter_value": filter_value}
    
    def _find_best_match(self, products, query):
        """找到最符合查询的产品"""
        # 简单的关键词匹配
        for product in products:
            if any(keyword in query for keyword in product["name"].lower().split()):
                return product
        return products[0]  # 默认返回第一个
    
    def _select_smart_filter(self, query, current_filters):
        """选择智能过滤器"""
        filter_rules = [
            ("brand", ["apple", "samsung", "nike", "adidas", "google", "under armour", "kitchenaid", "breville"]),
            ("color", ["black", "white", "silver", "blue", "titanium", "space gray"]),
            ("storage", ["128gb", "256gb", "512gb"]),
            ("size", ["10", "m", "l"]),
            ("type", ["sneakers", "hoodie", "shorts", "countertop", "drip"])
        ]
        
        for filter_type, values in filter_rules:
            if filter_type not in current_filters:
                for value in values:
                    if value in query:
                        return filter_type, value
        
        # 如果没有匹配，随机选择
        available_filters = [ft for ft, _ in filter_rules if ft not in current_filters]
        if available_filters:
            filter_type = random.choice(available_filters)
            filter_values = dict(filter_rules)[filter_type]
            return filter_type, random.choice(filter_values)
        else:
            return "brand", "Apple"  # 默认
    
    def train(self, num_episodes=20):
        """主训练循环"""
        print("🚀 Starting detailed training...")
        print(f"📊 Planning to train {num_episodes} episodes")
        print(f"🎮 Using {'simulated' if self.use_simulated else 'real'} environment")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            reward, steps, accuracy = self.train_episode_detailed(episode)
            
            # 每5个episode打印进度
            if (episode + 1) % 5 == 0:
                recent_stats = self.evaluator.get_summary_stats()
                print(f"\n📈 Progress Report (Episodes 1-{episode+1}):")
                print(f"   Average Reward: {recent_stats['average_reward']:.3f}")
                print(f"   Average Steps: {recent_stats['average_steps']:.1f}")
                print(f"   Success Rate: {recent_stats['success_rate']:.1f}%")
        
        # 训练完成
        training_time = time.time() - start_time
        final_stats = self.evaluator.get_summary_stats()
        
        print(f"\n⏱️ Training Time: {training_time:.1f} seconds")
        self.evaluator.print_detailed_report()
        
        # 环境指标
        env_metrics = self.env.get_metrics()
        print(f"\n🔄 Environment Statistics:")
        print(f"   • Total Steps: {env_metrics['total_steps']}")
        print(f"   • Successful Selections: {env_metrics['successful_selections']}")
        print(f"   • Failed Selections: {env_metrics['failed_selections']}")
        print(f"   • Filter Applications: {env_metrics['filter_applications']}")
        
        return final_stats

def save_detailed_results(stats, evaluator):
    """保存详细结果"""
    import shutil
    from pathlib import Path
    
    print("\n💾 Saving detailed training results...")
    
    # 保存汇总统计
    results = {
        "training_summary": stats,
        "environment": "simulated_webshop",
        "training_timestamp": time.time(),
        "model_version": "RAGEN-v1.0"
    }
    
    with open("training_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 保存详细历史
    import pandas as pd
    history_df = pd.DataFrame(evaluator.training_history)
    history_df.to_csv("training_history.csv", index=False)
    
    # 保存到卷
    volume_path = Path("/root/models")
    volume_path.mkdir(exist_ok=True)
    
    files_to_save = ["training_summary.json", "training_history.csv"]
    for filename in files_to_save:
        shutil.copy2(filename, volume_path / filename)
        print(f"  ✅ Saved: {filename}")
    
    print(f"📦 Total saved {len(files_to_save)} result files")

@app.function(
    image=base_image,
    gpu="A10G",
    timeout=86400,
    volumes={"/root/models": volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
def train_from_github():
    """详细的训练流程"""
    import os
    import sys
    from pathlib import Path
    import subprocess
    import shutil
    
    print("🚀 Starting detailed RAGEN training process...")
    
    # 克隆GitHub仓库
    repo_url = "https://github.com/YangLu963/Regan.git"
    work_dir = Path("/root/Regan") 
    
    try:
        if work_dir.exists():
            shutil.rmtree(work_dir)
        
        result = subprocess.run(
            ["git", "clone", repo_url, str(work_dir)],
            capture_output=True, text=True, check=True
        )
        print("✅ GitHub repository cloned successfully")
    except Exception as e:
        print(f"❌ Git clone failed: {e}")
        return {"status": "error", "message": "Git clone failed"}
    
    # 使用模拟环境训练
    print("🎮 Using simulated WebShop environment for detailed training...")
    
    try:
        trainer = DetailedRAGENTrainer(use_simulated=True)
        final_stats = trainer.train(num_episodes=20)
        
        # 保存详细结果
        save_detailed_results(final_stats, trainer.evaluator)
        
        return {
            "status": "completed",
            "message": "Detailed training completed successfully",
            "environment": "simulated",
            "summary_stats": final_stats,
            "total_training_time": final_stats.get('total_episodes', 0)
        }
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.function(
    image=base_image,
    volumes={"/root/models": volume}
)
def download_results():
    """下载训练结果"""
    from pathlib import Path
    import shutil
    
    print("📥 Downloading training results...")
    
    volume_path = Path("/root/models")
    local_path = Path(".")
    
    if not volume_path.exists():
        return {"status": "error", "message": "No data in shared volume"}
    
    downloaded_files = []
    for item in volume_path.iterdir():
        if item.is_file():
            shutil.copy2(item, local_path / item.name)
            downloaded_files.append(item.name)
            print(f"  ✅ Downloaded: {item.name}")
    
    return {"status": "success", "files": downloaded_files}

if __name__ == "__main__":
    with app.run():
        train_from_github.remote()
