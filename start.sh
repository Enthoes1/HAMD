#!/bin/bash

# 加载环境变量
set -a
source .env
set +a

# 创建必要的目录
mkdir -p ${HAMD_STORAGE_PATH}/assessment_results
mkdir -p ${HAMD_STORAGE_PATH}/progress
chmod -R 755 ${HAMD_STORAGE_PATH}

# 安装依赖
pip install --no-cache-dir -r requirements.txt

# 启动应用
python src/app.py