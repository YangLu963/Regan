# app.py
import modal

app = modal.App("ragen-github-webshop")

# WebShop专用镜像
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.37.0", 
        "accelerate>=0.24.1",
        "numpy>=1.24.3",
        "requests>=2.31.0",
        "PyYAML>=6.0.1", 
        "urllib3>=1.26.18",
        "tqdm>=4.66.1",
        "flask>=2.3.0",
        "flask-cors>=4.0.0", 
        "beautifulsoup4>=4.12.0",
        "scikit-learn",
        "pandas",
        "nmslib"
    )
    .run_commands(
        "apt-get update && apt-get install -y git build-essential cmake",
        "git config --global http.postBuffer 1048576000"
    )
)

volume = modal.Volume.from_name("ragen-models", create_if_missing=True)

@app.function(
    image=image,
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
    
    # ================== 启动真实WebShop服务器 ==================
    print("🛠️ 启动真实WebShop服务器...")
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
        
        # 验证克隆是否成功
        if not webshop_dir.exists():
            raise Exception("WebShop目录创建失败")
        
        # 检查目录内容
        print("🔍 检查WebShop目录内容...")
        result = subprocess.run(["ls", "-la"], cwd=str(webshop_dir), capture_output=True, text=True)
        print("WebShop目录内容:")
        print(result.stdout)
        
        # 2. 查找正确的启动方式（WebShop没有run.py）
        print("🔍 查找WebShop启动方式...")
        
        # 检查web_agent_site目录
        web_agent_dir = webshop_dir / "web_agent_site"
        if not web_agent_dir.exists():
            raise Exception("web_agent_site目录不存在")
        
        # 查看web_agent_site目录内容
        print("📁 web_agent_site目录内容:")
        result = subprocess.run(["ls", "-la"], cwd=str(web_agent_dir), capture_output=True, text=True)
        print(result.stdout)
        
        # 3. 安装WebShop依赖
        print("📦 安装WebShop依赖...")
        requirements_file = webshop_dir / "requirements.txt"
        if requirements_file.exists():
            subprocess.run([
                "pip", "install", "-r", str(requirements_file)
            ], check=True, timeout=180)
            print("✅ WebShop依赖安装完成")
        else:
            print("⚠️ 未找到requirements.txt，使用预安装依赖")

        # 4. 尝试多种启动方式
        print("🚀 尝试启动WebShop服务器...")
        server_started = False
        start_method = None
        
        # 可能的启动命令列表
        start_attempts = [
            {
                "name": "web_agent_site模块启动",
                "command": ["python", "-m", "web_agent_site.server"],
                "cwd": str(web_agent_dir)
            },
            {
                "name": "直接server.py启动", 
                "command": ["python", "server.py"],
                "cwd": str(web_agent_dir)
            },
            {
                "name": "shell脚本启动",
                "command": ["bash", "../run_web_agent_site_env.sh"],
                "cwd": str(webshop_dir)
            },
            {
                "name": "开发脚本启动",
                "command": ["bash", "../run_dev.sh"],
                "cwd": str(webshop_dir)
            }
        ]
        
        for attempt in start_attempts:
            print(f"🔄 尝试: {attempt['name']}")
            print(f"命令: {' '.join(attempt['command'])}")
            
            try:
                # 启动进程
                webshop_process = subprocess.Popen(
                    attempt['command'],
                    cwd=attempt['cwd'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # 等待并检查服务器状态
                for i in range(30):
                    try:
                        # 检查进程是否存活
                        if webshop_process.poll() is not None:
                            stdout, stderr = webshop_process.communicate()
                            print(f"❌ 进程退出 - {attempt['name']}:")
                            if stderr:
                                print(f"错误: {stderr[-500:]}")
                            break
                        
                        # 检查HTTP连接
                        response = requests.get("http://localhost:3000/", timeout=2)
                        if response.status_code == 200:
                            server_started = True
                            start_method = attempt['name']
                            print(f"✅ WebShop服务器启动成功！使用方式: {attempt['name']}")
                            break
                    except requests.exceptions.ConnectionError:
                        pass
                    except Exception as e:
                        if i % 10 == 0:
                            print(f"⏳ 等待中... ({i+1}/30)")
                    
                    time.sleep(1)
                
                if server_started:
                    break
                else:
                    # 终止当前进程，尝试下一个
                    if webshop_process and webshop_process.poll() is None:
                        webshop_process.terminate()
                        webshop_process.wait(timeout=5)
                    webshop_process = None
                    
            except Exception as e:
                print(f"⚠️ 启动方式 {attempt['name']} 失败: {e}")
                continue
        
        if not server_started:
            print("❌ 所有启动方式都失败，WebShop服务器启动失败")
            # 尝试获取最后的错误信息
            if webshop_process:
                try:
                    stdout, stderr = webshop_process.communicate(timeout=5)
                    if stderr:
                        print(f"最后错误信息: {stderr[-1000:]}")
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
    image=image,
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

# 调试函数
@app.function(image=image)
def debug_webshop():
    """调试WebShop安装和启动"""
    import subprocess
    from pathlib import Path
    import shutil
    import requests
    import time
    
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
    
    # 检查web_agent_site目录
    web_agent_dir = webshop_dir / "web_agent_site"
    if web_agent_dir.exists():
        print("📁 web_agent_site目录内容:")
        result = subprocess.run(["ls", "-la"], cwd=str(web_agent_dir), capture_output=True, text=True)
        print(result.stdout)
    else:
        print("❌ web_agent_site目录不存在")
        return {"status": "error", "message": "web_agent_site目录不存在"}
    
    # 安装依赖
    requirements_file = webshop_dir / "requirements.txt"
    if requirements_file.exists():
        print("📦 安装requirements.txt...")
        result = subprocess.run([
            "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True, timeout=180)
        if result.returncode == 0:
            print("✅ 依赖安装成功")
        else:
            print(f"⚠️ 依赖安装问题: {result.stderr}")
    
    # 尝试启动
    print("🚀 尝试启动WebShop...")
    process = subprocess.Popen(
        ["python", "-m", "web_agent_site.server"],
        cwd=str(web_agent_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 等待并检查
    time.sleep(15)  # 等待15秒
    
    # 检查进程状态
    if process.poll() is None:
        print("✅ WebShop进程正在运行")
        # 测试连接
        try:
            response = requests.get("http://localhost:3000/", timeout=5)
            print(f"✅ 服务器响应: {response.status_code}")
        except Exception as e:
            print(f"❌ 连接失败: {e}")
        process.terminate()
    else:
        stdout, stderr = process.communicate()
        print(f"❌ 进程退出:")
        if stdout:
            print(f"STDOUT: {stdout[-1000:]}")
        if stderr:
            print(f"STDERR: {stderr[-1000:]}")
    
    return {"status": "debug_complete"}

if __name__ == "__main__":
    with app.run():
        train_from_github.remote()
