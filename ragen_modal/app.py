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
        "urllib3>=1.26.18",
        "tqdm>=4.66.1"
    )
    .run_commands(
        "apt-get update && apt-get install -y git build-essential cmake",
        "git config --global http.postBuffer 1048576000"
    )
)

# 第二阶段：WebShop专用镜像（预编译所有依赖）
webshop_image = base_image.pip_install(
    "flask>=2.3.0",
    "flask-cors>=4.0.0", 
    "beautifulsoup4>=4.12.0",
    "scikit-learn",
    "pandas",
    # nmslib 单独处理，避免编译超时
)

volume = modal.Volume.from_name("ragen-models", create_if_missing=True)

@app.function(
    image=webshop_image,  # 使用预构建的WebShop镜像
    gpu="A10G",
    timeout=86400,
    volumes={"/root/models": volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
def train_from_github():
    """从GitHub克隆项目并使用真实WebShop训练"""
    import os
    import sys
    from pathlib import Path
    import subprocess
    import time
    import requests
    
    print("🚀 从GitHub克隆RAGEN项目...")
    
    # 克隆你的GitHub仓库
    repo_url = "https://github.com/YangLu963/Regan.git"
    work_dir = Path("/root/ragen_project")
    
    try:
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
    
    # ================== 启动真实WebShop服务器 ==================
    print("🛠️ 启动真实WebShop服务器...")
    webshop_process = None
    
    try:
        # 1. 克隆官方WebShop仓库
        webshop_dir = Path("/root/WebShop")
        if not webshop_dir.exists():
            print("📥 克隆WebShop官方仓库...")
            subprocess.run([
                "git", "clone", "https://github.com/princeton-nlp/WebShop.git", 
                str(webshop_dir)
            ], check=True, timeout=120)
            print("✅ WebShop仓库克隆完成")
        
        # 2. 安装nmslib（单独处理，避免超时）
        print("📦 安装nmslib...")
        try:
            # 尝试快速安装
            subprocess.run([
                "pip", "install", "nmslib"
            ], check=True, timeout=300)  # 5分钟超时
            print("✅ nmslib安装成功")
        except subprocess.TimeoutExpired:
            print("⚠️ nmslib安装超时，尝试跳过...")
        except Exception as e:
            print(f"⚠️ nmslib安装失败: {e}")
        
        # 3. 检查并安装其他可能缺失的依赖
        print("🔍 检查WebShop依赖...")
        try:
            # 测试导入WebShop关键模块
            import flask
            import flask_cors
            import bs4
            print("✅ WebShop核心依赖已就绪")
        except ImportError as e:
            print(f"⚠️ 依赖缺失: {e}")
            print("📦 安装缺失依赖...")
            subprocess.run([
                "pip", "install", "flask", "flask-cors", "beautifulsoup4"
            ], check=True, timeout=60)
        
        # 4. 启动WebShop服务器
        print("🚀 启动WebShop服务进程...")
        webshop_process = subprocess.Popen([
            "python", "run.py", "--port", "3000"
        ], cwd=str(webshop_dir), 
           stdout=subprocess.PIPE, 
           stderr=subprocess.PIPE,
           text=True)
        
        # 5. 等待服务器启动（更详细的检查）
        print("⏳ 等待WebShop服务器启动...")
        server_started = False
        
        for i in range(45):  # 增加到45秒
            try:
                # 检查进程是否存活
                if webshop_process.poll() is not None:
                    # 进程已结束，读取错误输出
                    stdout, stderr = webshop_process.communicate()
                    print(f"❌ WebShop进程异常退出:")
                    if stdout:
                        print(f"STDOUT: {stdout[-500:]}")  # 最后500字符
                    if stderr:
                        print(f"STDERR: {stderr[-500:]}")
                    break
                
                # 检查HTTP连接
                response = requests.get("http://localhost:3000/", timeout=5)
                if response.status_code == 200:
                    server_started = True
                    print("✅ WebShop服务器启动成功！")
                    break
                else:
                    if i % 10 == 0:  # 每10次打印一次
                        print(f"⏳ 服务器状态码 {response.status_code}，继续等待... ({i+1}/45)")
            except requests.exceptions.ConnectionError:
                if i % 10 == 0:
                    print(f"⏳ 连接拒绝，继续等待... ({i+1}/45)")
            except Exception as e:
                if i % 10 == 0:
                    print(f"⏳ 等待中... ({i+1}/45) - {str(e)[:100]}")
            
            time.sleep(1)
        
        if not server_started:
            print("❌ WebShop服务器启动失败")
            # 尝试读取进程输出获取更多信息
            try:
                stdout, stderr = webshop_process.communicate(timeout=5)
                if stdout:
                    print(f"最后输出: {stdout[-1000:]}")
                if stderr:
                    print(f"错误信息: {stderr[-1000:]}")
            except:
                pass
            return {"status": "error", "message": "WebShop服务器启动失败"}
        else:
            print("🎯 真实WebShop环境准备就绪！")
            os.environ["USE_SIMULATED_WEBSHOP"] = "false"
            
    except Exception as e:
        print(f"⚠️ WebShop服务器启动过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"WebShop启动失败: {str(e)}"}
    
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
        
        return {
            "status": "completed", 
            "message": "训练成功完成",
            "github_repo": repo_url,
            "webshop_mode": "real"
        }
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        
        # 确保服务器被停止
        if webshop_process:
            webshop_process.terminate()
        
        return {"status": "error", "message": str(e)}

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

if __name__ == "__main__":
    with app.run():
        train_from_github.remote()
