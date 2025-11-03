# app.py
import modal

app = modal.App("ragen-github")

# 镜像配置 - 包含git和所有依赖
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.0.1",
        "transformers==4.35.0", 
        "accelerate==0.24.1",
        "numpy==1.24.3",
        "requests==2.31.0",
        "PyYAML==6.0.1",
        "urllib3==1.26.18",
        "tqdm==4.66.1"
    )
    .run_commands(
        "apt-get update && apt-get install -y git",
        "git config --global http.postBuffer 1048576000"
    )
)

# 共享卷用于保存结果
volume = modal.Volume.from_name("ragen-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,  # 24小时
    volumes={"/root/models": volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
def train_from_github():
    """从GitHub克隆项目并训练"""
    import os
    import sys
    from pathlib import Path
    import subprocess
    
    print("🚀 从GitHub克隆RAGEN项目...")
    
    # 克隆你的GitHub仓库
    repo_url = "https://github.com/YangLu963/Regan.git"
    work_dir = Path("/root/ragen_project")
    
    try:
        # 克隆仓库
        result = subprocess.run(
            ["git", "clone", repo_url, str(work_dir)],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ GitHub仓库克隆成功")
    except subprocess.CalledProcessError as e:
        print(f"❌ Git克隆失败: {e}")
        return {"status": "error", "message": "Git克隆失败"}
    
    # 切换到项目子目录 ragen_modal
    project_dir = work_dir / "ragen_modal"
    os.chdir(project_dir)
    sys.path.insert(0, str(project_dir))  # 添加到Python路径开头
    
    # 显示项目结构和调试信息
    print("📁 项目文件结构:")
    for item in project_dir.rglob("*"):
        if item.is_file() and not any(part.startswith('.') for part in item.parts):
            print(f"  📄 {item.relative_to(project_dir)}")
    
    print(f"🔍 当前工作目录: {os.getcwd()}")
    print(f"🔍 Python路径: {sys.path}")
    
    try:
        # 导入并运行训练器
        print("\n🎯 导入训练模块...")
        from ragen.train_ragen_apo import RAGENWebShopTrainer
        
        print("🚀 开始训练...")
        trainer = RAGENWebShopTrainer()
        trainer.train()
        
        # 保存结果到卷
        save_results_to_volume()
        
        return {
            "status": "completed", 
            "message": "训练成功完成",
            "github_repo": repo_url
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

if __name__ == "__main__":
    with app.run():
        train_from_github.remote()
