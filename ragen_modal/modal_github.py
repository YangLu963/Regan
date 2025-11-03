import modal

app = modal.App("ragen-github")

# 从GitHub克隆完整项目
image = modal.Image.debian_slim().pip_install([
    "torch", "transformers", "numpy", "pyyaml", "requests", "accelerate"
]).run_commands([
    "git clone https://github.com/YangLu963/Regan.git /root/ragen",
    "cd /root/ragen"
])

@app.function(image=image, gpu="A10G", timeout=3600)
def train():
    import os
    os.chdir("/root/ragen")
    print("🚀 开始RAGEN训练...")
    os.system("python train_ragen_apo.py")

if __name__ == "__main__":
    with app.run():
        train.remote()
