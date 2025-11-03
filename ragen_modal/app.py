# app.py
import modal
import os
import sys
from pathlib import Path

app = modal.App("ragen-webshop-trainer")

# 基础镜像
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
)

# 共享卷用于保存模型
volume = modal.Volume.from_name("ragen-models", create_if_missing=True)

@app.function(
    image=image,
    gpu="A10G",
    timeout=86400,
    volumes={"/root/models": volume},
    secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
def train_ragen():
    """在Modal上训练RAGEN - 直接使用你的现有代码"""
    import torch
    import yaml
    
    print("🚀 开始在Modal上训练RAGEN...")
    print("=" * 50)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 50)
    
    # 设置工作目录 - 使用git克隆或手动上传文件
    work_dir = Path("/root/ragen_project")
    work_dir.mkdir(exist_ok=True)
    os.chdir(work_dir)
    
    # 由于Mount有问题，我们需要手动复制文件
    copy_project_files()
    
    sys.path.append(str(work_dir))
    
    try:
        # 导入并运行训练器
        print("\n🎯 导入训练模块...")
        from train_ragen_apo import RAGENWebShopTrainer
        
        print("🚀 开始训练...")
        trainer = RAGENWebShopTrainer()
        trainer.train()
        
        # 保存结果
        save_results_to_volume()
        
        return {
            "status": "completed", 
            "message": "训练成功完成",
            "gpu_used": torch.cuda.get_device_name() if torch.cuda.is_available() else "None"
        }
        
    except Exception as e:
        print(f"❌ 训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

def copy_project_files():
    """手动复制项目文件（因为Mount有问题）"""
    import shutil
    from pathlib import Path
    
    print("📁 设置项目文件...")
    
    # 创建必要的目录结构
    directories = ["ragen", "configs", "logs"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    
    # 这里我们假设文件已经通过其他方式上传
    # 在实际部署时，你可能需要手动上传文件或使用git
    
    print("✅ 项目目录结构创建完成")

def save_results_to_volume():
    """保存训练结果到共享卷"""
    import shutil
    from pathlib import Path
    
    print("\n💾 保存训练结果...")
    
    saved_files = []
    patterns = ["*.pth", "*.pt", "*.bin", "*.yaml", "*.yml", "*.json", "*.log"]
    
    for pattern in patterns:
        for file_path in Path(".").glob(pattern):
            if file_path.is_file():
                dest_path = Path("/root/models") / file_path.name
                shutil.copy2(file_path, dest_path)
                saved_files.append(file_path.name)
                print(f"  ✅ 保存: {file_path.name}")
    
    print(f"📦 总共保存了 {len(saved_files)} 个文件到共享卷")

@app.function(
    image=image,
    volumes={"/root/models": volume}
)
def download_results():
    """下载训练结果到本地"""
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
    
    return {
        "status": "success", 
        "downloaded_files": downloaded_files,
        "count": len(downloaded_files)
    }

@app.function(image=image)
def check_environment():
    """检查Modal环境"""
    import torch
    import importlib
    
    print("🔍 检查Modal环境...")
    
    # 检查GPU
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "None",
    }
    
    # 检查关键包
    packages = ["torch", "transformers", "numpy", "yaml", "requests"]
    package_versions = {}
    for package in packages:
        try:
            mod = importlib.import_module(package)
            package_versions[package] = getattr(mod, "__version__", "Unknown")
        except ImportError:
            package_versions[package] = "Not installed"
    
    return {
        "gpu": gpu_info,
        "packages": package_versions
    }

if __name__ == "__main__":
    # 直接运行训练
    with app.run():
        result = train_ragen.remote()
        print(f"训练结果: {result}")