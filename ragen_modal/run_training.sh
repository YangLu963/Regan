# run_training.sh
#!/bin/bash

echo "🏃 快速启动RAGEN训练..."
echo "这将使用A10G GPU训练24小时"

# 运行训练
modal run app.py::train_ragen

echo ""
echo "🎯 训练任务已提交!"
echo "使用以下命令查看进度:"
echo "  modal logs ragen-webshop-trainer"