cat > deploy_github.sh << 'EOF'
#!/bin/bash

echo "🚀 部署GitHub版本的RAGEN训练系统..."
echo "仓库: https://github.com/YangLu963/Regan"

# 检查环境变量
if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "❌ 请设置HUGGINGFACE_TOKEN环境变量"
    exit 1
fi

# 部署
modal deploy app_github.py

echo ""
echo "✅ 部署完成!"
echo ""
echo "📋 使用命令:"
echo "   modal run app_github.py::train_from_github    # 运行训练"
echo "   modal run app_github.py::download_results     # 下载结果"
EOF

chmod +x deploy_github.sh
