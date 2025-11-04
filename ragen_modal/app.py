# app.py
import modal

app = modal.App("ragen-github-webshop")

# 第一阶段：基础镜像（只包含必要依赖）
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.37.0", 
        "accelerate>=0.24.1",
        "numpy>=1.24.3",
        "requests>=2.31.0",
        "PyYAML>=6.0.1", 
        "urllib3>=2.0.0",  # 保持高版本
        "tqdm>=4.66.1",
        "flask>=2.3.0",
        "flask-cors>=4.0.0",
        "beautifulsoup4>=4.12.0"
    )
    .run_commands(
        "apt-get update && apt-get install -y git build-essential cmake",
        "git config --global http.postBuffer 1048576000"
    )
)

volume = modal.Volume.from_name("ragen-models", create_if_missing=True)

@app.function(
    image=base_image,
    gpu="A10G",
    timeout=86400,
    volumes={"/root/models": volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
def train_from_github():
    """从GitHub克隆项目并使用WebShop训练"""
    import os
    import sys
    from pathlib import Path
    import subprocess
    import time
    import requests
    import shutil
    
    print("🚀 从GitHub克隆RAGEN项目...")
    
    # 克隆你的GitHub仓库
    repo_url = "https://github.com/YangLu963/Regan.git"
    work_dir = Path("/root/ragen_project")
    
    try:
        # 清理旧目录
        if work_dir.exists():
            shutil.rmtree(work_dir)
        
        result = subprocess.run(
            ["git", "clone", repo_url, str(work_dir)],
            capture_output=True, text=True, check=True
        )
        print("✅ GitHub仓库克隆成功")
    except subprocess.CalledProcessError as e:
        print(f"❌ Git克隆失败: {e}")
        return {"status": "error", "message": "Git克隆失败"}
    
    project_dir = work_dir / "ragen_modal"
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir))
    
    # ================== 启动WebShop服务器 ==================
    print("🛠️ 启动WebShop服务器...")
    webshop_process = None
    
    try:
        # 1. 克隆官方WebShop仓库
        webshop_dir = Path("/root/WebShop")
        
        # 清理旧目录
        if webshop_dir.exists():
            shutil.rmtree(webshop_dir)
            print("🗑️ 清理旧WebShop目录")
        
        print("📥 克隆WebShop官方仓库...")
        result = subprocess.run([
            "git", "clone", "https://github.com/princeton-nlp/WebShop.git", 
            str(webshop_dir)
        ], capture_output=True, text=True, check=True, timeout=180)
        
        print("✅ WebShop仓库克隆完成")
        
        # 检查目录结构
        print("📁 WebShop目录结构:")
        result = subprocess.run(["find", ".", "-name", "*.py", "-type", "f"], 
                              cwd=str(webshop_dir), capture_output=True, text=True)
        print(result.stdout[:2000])  # 显示更多内容
        
        # 2. 查找正确的启动文件
        possible_start_files = [
            "run.py",
            "server.py", 
            "web_agent_site/server.py",
            "web_agent_site/app.py",
            "src/server.py"
        ]
        
        start_file = None
        for file in possible_start_files:
            if (webshop_dir / file).exists():
                start_file = file
                print(f"✅ 找到启动文件: {file}")
                break
        
        if not start_file:
            print("❌ 未找到标准启动文件，使用模拟WebShop...")
            webshop_process = create_simulated_webshop()
            os.environ["USE_SIMULATED_WEBSHOP"] = "true"
        else:
            # 3. 安装WebShop特定依赖（避免冲突）
            print("📦 安装WebShop最小依赖...")
            webshop_deps = ["beautifulsoup4", "nmslib", "scikit-learn", "pandas", "flask", "flask-cors"]
            for dep in webshop_deps:
                try:
                    subprocess.run(["pip", "install", dep], check=True, timeout=60)
                    print(f"✅ 安装 {dep} 成功")
                except Exception as e:
                    print(f"⚠️ 安装 {dep} 失败: {e}")
            
            # 4. 启动WebShop服务器
            print(f"🚀 启动WebShop服务: {start_file}")
            webshop_process = subprocess.Popen([
                "python", start_file, "--port", "3000"
            ], cwd=str(webshop_dir), 
               stdout=subprocess.PIPE, 
               stderr=subprocess.PIPE,
               text=True)
            os.environ["USE_SIMULATED_WEBSHOP"] = "false"

        # 5. 等待服务器启动
        print("⏳ 等待WebShop服务器启动...")
        server_started = False
        
        for i in range(30):  # 30秒超时
            try:
                # 检查进程是否存活
                if webshop_process and webshop_process.poll() is not None:
                    stdout, stderr = webshop_process.communicate()
                    print(f"❌ WebShop进程异常退出:")
                    print(f"STDOUT: {stdout}")
                    print(f"STDERR: {stderr}")
                    
                    # 如果标准WebShop失败，回退到模拟版本
                    if not os.environ.get("USE_SIMULATED_WEBSHOP") == "true":
                        print("🔄 回退到模拟WebShop...")
                        webshop_process = create_simulated_webshop()
                        os.environ["USE_SIMULATED_WEBSHOP"] = "true"
                    break
                
                # 检查HTTP连接
                response = requests.get("http://localhost:3000/", timeout=5)
                if response.status_code == 200:
                    server_started = True
                    webshop_mode = "模拟" if os.environ.get("USE_SIMULATED_WEBSHOP") == "true" else "真实"
                    print(f"✅ {webshop_mode}WebShop服务器启动成功！")
                    break
                else:
                    if i % 5 == 0:
                        print(f"⏳ 服务器状态码 {response.status_code}，继续等待... ({i+1}/30)")
            except requests.exceptions.ConnectionError:
                if i % 5 == 0:
                    print(f"⏳ 连接拒绝，继续等待... ({i+1}/30)")
            except Exception as e:
                if i % 5 == 0:
                    print(f"⏳ 等待中... ({i+1}/30) - {str(e)[:100]}")
            
            time.sleep(1)
        
        if not server_started:
            print("❌ WebShop服务器启动失败，使用模拟环境继续训练")
            # 即使服务器启动失败，也继续训练（使用模拟环境）
            os.environ["USE_SIMULATED_WEBSHOP"] = "true"
            
    except Exception as e:
        print(f"⚠️ WebShop服务器启动过程中出错: {e}")
        print("🔄 使用模拟WebShop环境继续训练...")
        os.environ["USE_SIMULATED_WEBSHOP"] = "true"
        import traceback
        traceback.print_exc()
    
    # ================== 开始训练 ==================
    print("📁 项目文件结构:")
    for item in project_dir.rglob("*"):
        if item.is_file() and not any(part.startswith('.') for part in item.parts):
            print(f"  📄 {item.relative_to(project_dir)}")
    
    try:
        # 导入并运行训练器
        print("\n🎯 导入训练模块...")
        from ragen.train_ragen_apo import RAGENWebShopTrainer
        
        print("🚀 开始训练...")
        trainer = RAGENWebShopTrainer()
        trainer.train()
        
        # 保存结果到卷
        save_results_to_volume()
        
        # 训练完成后停止WebShop服务器
        if webshop_process:
            webshop_process.terminate()
            webshop_process.wait()
            print("🛑 WebShop服务器已停止")
        
        webshop_mode = "模拟" if os.environ.get("USE_SIMULATED_WEBSHOP") == "true" else "真实"
        return {
            "status": "completed", 
            "message": "训练成功完成",
            "github_repo": repo_url,
            "webshop_mode": webshop_mode
        }
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 确保服务器被停止
        if webshop_process:
            webshop_process.terminate()
        
        return {"status": "error", "message": str(e)}

def create_simulated_webshop():
    """创建模拟WebShop服务器"""
    print("🎭 创建模拟WebShop服务器...")
    
    server_code = '''
from flask import Flask, jsonify, request
import random
import time

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"status": "ready", "message": "WebShop Simulator", "mode": "simulated"})

@app.route('/search/<query>')
def search(query):
    """模拟搜索功能"""
    time.sleep(0.1)  # 模拟延迟
    
    products = []
    if "red" in query.lower() and "shirt" in query.lower():
        products = [
            {"id": "1001", "name": "Red Cotton T-Shirt", "price": 29.99, "color": "red", "size": "M"},
            {"id": "1002", "name": "Red Polo Shirt", "price": 39.99, "color": "red", "size": "L"}
        ]
    elif "black" in query.lower() and "backpack" in query.lower():
        products = [
            {"id": "2001", "name": "Black Laptop Backpack", "price": 49.99, "has_laptop_compartment": True},
            {"id": "2002", "name": "Black Travel Backpack", "price": 59.99, "has_laptop_compartment": True}
        ]
    else:
        products = [
            {"id": "3001", "name": "Blue Jeans", "price": 39.99, "color": "blue"},
            {"id": "3002", "name": "White Sneakers", "price": 59.99, "color": "white"}
        ]
    
    return jsonify({"products": products, "query": query})

@app.route('/click/<product_id>')
def click(product_id):
    """模拟点击商品"""
    time.sleep(0.1)
    
    product_details = {
        "1001": {"id": "1001", "name": "Red Cotton T-Shirt", "price": 29.99, "color": "red", "description": "Comfortable cotton t-shirt", "in_stock": True},
        "1002": {"id": "1002", "name": "Red Polo Shirt", "price": 39.99, "color": "red", "description": "Classic polo shirt", "in_stock": True},
        "2001": {"id": "2001", "name": "Black Laptop Backpack", "price": 49.99, "has_laptop_compartment": True, "description": "Durable laptop backpack", "in_stock": True},
        "2002": {"id": "2002", "name": "Black Travel Backpack", "price": 59.99, "has_laptop_compartment": True, "description": "Spacious travel backpack", "in_stock": True}
    }
    
    product = product_details.get(product_id, {"id": product_id, "name": "Unknown Product", "in_stock": False})
    return jsonify(product)

@app.route('/buy/<product_id>')
def buy(product_id):
    """模拟购买功能"""
    time.sleep(0.2)
    
    if product_id in ["1001", "1002", "2001", "2002"]:
        return jsonify({
            "success": True,
            "order_id": f"ORDER_{random.randint(1000,9999)}",
            "product_id": product_id,
            "message": "Purchase successful!"
        })
    else:
        return jsonify({
            "success": False,
            "error": "Product not found"
        }), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)
'''
    
    # 写入模拟服务器文件
    import tempfile
    temp_dir = tempfile.mkdtemp()
    server_file = Path(temp_dir) / "simulated_webshop.py"
    
    with open(server_file, 'w') as f:
        f.write(server_code)
    
    # 启动模拟服务器
    webshop_process = subprocess.Popen([
        "python", "simulated_webshop.py"
    ], cwd=temp_dir,
       stdout=subprocess.PIPE,
       stderr=subprocess.PIPE,
       text=True)
    
    return webshop_process

def save_results_to_volume():
    """保存训练结果到共享卷"""
    import shutil
    from pathlib import Path
    
    print("\n💾 保存训练结果...")
    saved_files = []
    patterns = ["*.pth", "*.pt", "*.bin", "*.yaml", "*.json", "*.log", "vstar_cache.pkl"]
    
    for pattern in patterns:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                dest_path = Path("/root/models") / file_path.name
                shutil.copy2(file_path, dest_path)
                saved_files.append(file_path.name)
                print(f"  ✅ 保存: {file_path.name}")
    
    print(f"📦 总共保存了 {len(saved_files)} 个文件")

@app.function(
    image=base_image,
    volumes={"/root/models": volume}
)
def download_results():
    """下载训练结果"""
    from pathlib import Path
    import shutil
    
    print("📥 下载训练结果...")
    
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

# 简化调试函数
@app.function(image=base_image)
def debug_webshop():
    """调试WebShop安装"""
    import subprocess
    from pathlib import Path
    import shutil
    
    print("🔧 调试WebShop安装...")
    
    webshop_dir = Path("/root/WebShop")
    
    # 清理旧目录
    if webshop_dir.exists():
        shutil.rmtree(webshop_dir)
    
    # 克隆WebShop
    print("📥 克隆WebShop...")
    result = subprocess.run([
        "git", "clone", "https://github.com/princeton-nlp/WebShop.git", 
        str(webshop_dir)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Git克隆失败: {result.stderr}")
        return {"status": "error", "message": "Git克隆失败"}
    
    print("✅ 克隆成功")
    
    # 检查目录内容
    print("📁 目录内容:")
    result = subprocess.run(["ls", "-la"], cwd=str(webshop_dir), capture_output=True, text=True)
    print(result.stdout)
    
    # 查找启动文件
    print("🔍 查找启动文件...")
    result = subprocess.run(["find", ".", "-name", "*.py", "-type", "f"], 
                          cwd=str(webshop_dir), capture_output=True, text=True)
    print(result.stdout)
    
    return {"status": "debug_complete", "message": "检查完成"}

if __name__ == "__main__":
    with app.run():
        train_from_github.remote()
