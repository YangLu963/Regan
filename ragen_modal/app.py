import modal

app = modal.App("ragen-github-webshop")

# 第一阶段：基础镜像（只包含必要依赖）
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
        "urllib3>=2.0.0",  # 保持高版本
        "tqdm>=4.66.1",
        "flask>=2.3.0",
        "flask-cors>=4.0.0"
        )  
      .run_commands(
        "git config --global http.postBuffer 1048576000"
    )
)

# 第二阶段：WebShop专用镜像（隔离环境）
webshop_image = base_image.run_commands(
    # 创建WebShop专用虚拟环境
    "python -m venv /root/webshop_venv",
    # 激活虚拟环境安装WebShop依赖
    ". /root/webshop_venv/bin/activate && pip install beautifulsoup4 nmslib scikit-learn pandas"
)

volume = modal.Volume.from_name("ragen-models", create_if_missing=True)

def manual_webshop_start(webshop_dir):
    """手动启动WebShop的备选方案"""
    print("🛠️ 使用手动启动方案...")
    # 这里可以添加手动启动逻辑
    return {"status": "manual_start", "message": "使用手动启动"}

@app.function(
    image=webshop_image,
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
    work_dir = Path("/root/Regan") 
    
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
    
    print("🔧 修正WebShop启动方式...")
    
    # 1. 检查WebShop实际目录结构
    webshop_dir = Path("/root/WebShop")
    print("📁 WebShop目录结构:")
    result = subprocess.run(["find", ".", "-name", "*.py", "-type", "f"], 
                          cwd=str(webshop_dir), capture_output=True, text=True)
    print(result.stdout)
    
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
        print("❌ 未找到启动文件，尝试手动启动...")
        # 手动启动方案
        return manual_webshop_start(webshop_dir)
    
    # 3. 使用虚拟环境启动
    print("🚀 使用虚拟环境启动WebShop...")
    webshop_process = subprocess.Popen([
        "/root/webshop_venv/bin/python", start_file, "--port", "3000"
    ], cwd=str(webshop_dir), 
       stdout=subprocess.PIPE, 
       stderr=subprocess.PIPE,
       text=True)

    # 4. 等待服务器启动
    print("⏳ 等待WebShop服务器启动...")
    server_started = False
    
    for i in range(60):  # 增加到60秒
        try:
            # 检查进程是否存活
            if webshop_process.poll() is not None:
                stdout, stderr = webshop_process.communicate()
                print(f"❌ WebShop进程异常退出:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                
                # 尝试诊断问题
                if "No module named" in stderr:
                    print("🔧 检测到模块缺失，尝试安装依赖...")
                    # 安装WebShop特定依赖
                    requirements_file = webshop_dir / "requirements.txt"
                    if requirements_file.exists():
                        subprocess.run([
                            "pip", "install", "-r", str(requirements_file)
                        ], check=True, timeout=120)
                        print("✅ 依赖安装完成，重新启动...")
                        # 重新启动
                        webshop_process = subprocess.Popen([
                            "python", "run.py", "--port", "3000"
                        ], cwd=str(webshop_dir), 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE,
                           text=True)
                        continue
                break
            
            # 检查HTTP连接
            response = requests.get("http://localhost:3000/", timeout=5)
            if response.status_code == 200:
                server_started = True
                print("✅ WebShop服务器启动成功！")
                break
            else:
                if i % 10 == 0:
                    print(f"⏳ 服务器状态码 {response.status_code}，继续等待... ({i+1}/60)")
        except requests.exceptions.ConnectionError:
            if i % 10 == 0:
                print(f"⏳ 连接拒绝，继续等待... ({i+1}/60)")
        except Exception as e:
            if i % 10 == 0:
                print(f"⏳ 等待中... ({i+1}/60) - {str(e)[:100]}")
        
        time.sleep(1)
    
    if not server_started:
        print("❌ WebShop服务器启动失败")
        return {"status": "error", "message": "WebShop服务器启动失败"}
    else:
        print("🎯 真实WebShop环境准备就绪！")
        os.environ["USE_SIMULATED_WEBSHOP"] = "false"
    
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

# 添加调试函数
@app.function(image=webshop_image)
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
    
    # 检查run.py
    run_py = webshop_dir / "run.py"
    print(f"run.py存在: {run_py.exists()}")
    
    if run_py.exists():
        # 尝试安装requirements
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
            ["python", "run.py", "--port", "3000"], 
            cwd=str(webshop_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        import time
        time.sleep(10)  # 等待10秒
        
        # 检查进程状态
        if process.poll() is None:
            print("✅ WebShop进程正在运行")
            # 测试连接
            try:
                import requests
                response = requests.get("http://localhost:3000/", timeout=5)
                print(f"✅ 服务器响应: {response.status_code}")
            except Exception as e:
                print(f"❌ 连接失败: {e}")
            process.terminate()
        else:
            stdout, stderr = process.communicate()
            print(f"❌ 进程退出:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
    
    return {"status": "debug_complete"}

if __name__ == "__main__":
    with app.run():
        train_from_github.remote()
