# deploy.sh
#!/bin/bash

echo "🚀 部署RAGEN训练系统到Modal..."
echo "=========================================="

# 1. 检查HuggingFace token
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "❌ 请设置HUGGINGFACE_TOKEN环境变量"
    echo "运行: export HUGGINGFACE_TOKEN=你的_hf_token"
    echo "可以在 https://huggingface.co/settings/tokens 获取token"
    exit 1
fi

# 2. 检查Modal安装
if ! command -v modal &> /dev/null; then
    echo "❌ 请先安装Modal: pip install modal"
    exit 1
fi

# 3. 创建Modal secret
echo "🔐 创建Modal secret..."
modal secret create my-huggingface-secret HUGGINGFACE_TOKEN=$HUGGINGFACE_TOKEN

# 4. 部署应用
echo "📦 部署应用到Modal..."
modal deploy app.py

echo ""
echo "✅ 部署完成!"
echo "=========================================="
echo ""
echo "📋 使用以下命令:"
echo "   modal run app.py::train_ragen        # 运行训练"
echo "   modal run app.py::download_results   # 下载结果" 
echo "   modal run app.py::check_environment  # 检查环境"
echo "   modal run app.py::develop            # 开发模式"
echo ""
echo "🔍 监控训练进度:"
echo "   modal logs ragen-webshop-trainer"
echo ""
echo "💾 查看保存的模型:"
echo "   modal volume ls ragen-models"