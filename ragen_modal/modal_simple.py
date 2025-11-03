import modal

app = modal.App("ragen-simple")

image = modal.Image.debian_slim().pip_install([
    "torch", "transformers", "numpy", "pyyaml", "requests", "accelerate"
])

@app.function(image=image, gpu="A10G", timeout=3600)
def train():
    # 直接导入训练类（Modal会自动上传所有import的文件）
    from train_ragen_apo import RAGENWebShopTrainer
    trainer = RAGENWebShopTrainer()
    trainer.train()

@app.local_entrypoint()
def main():
    train.remote()

if __name__ == "__main__":
    main()