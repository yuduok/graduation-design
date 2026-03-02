#!/bin/bash
# 启动Streamlit演示界面

# 使用CoOp的虚拟环境
source "$(dirname "$0")/../CoOp/venv/bin/activate"

cd "$(dirname "$0")/demo"

echo "启动细粒度猫狗分类演示界面..."
echo "访问 http://localhost:8501"
echo ""

pip install streamlit flask flask-cors matplotlib seaborn scikit-learn -q

streamlit run pet_classifier_demo.py
