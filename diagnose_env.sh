#!/bin/bash
# 环境诊断脚本

echo "=========================================="
echo "环境诊断"
echo "=========================================="
echo ""

echo "1. Python 路径和版本:"
which python
python --version
echo ""

echo "2. Conda 环境:"
conda env list | grep "*"
echo ""

echo "3. 检查关键包:"
echo -n "torch: "
python -c "import torch; print(torch.__version__)" 2>&1 || echo "未安装"

echo -n "dgl: "
python -c "import dgl; print(dgl.__version__)" 2>&1 || echo "未安装 ❌"

echo -n "transformers: "
python -c "import transformers; print(transformers.__version__)" 2>&1 || echo "未安装"

echo ""
echo "4. 尝试导入 config:"
python -c "from config import TrainingConfig; print('✅ config.py 导入成功')" 2>&1

echo ""
echo "5. 尝试导入 alignn_cgcnn:"
python -c "from models.alignn_cgcnn import ACGCNNConfig; print('✅ alignn_cgcnn 导入成功')" 2>&1 || echo "❌ alignn_cgcnn 导入失败（这是正常的，如果 dgl 未安装）"

echo ""
echo "=========================================="
echo "诊断完成"
echo "=========================================="
