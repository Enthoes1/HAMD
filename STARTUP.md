# HAMD 项目启动指南

## 环境要求
- Python 3.x
- Linux/Windows 系统
- DashScope API Key

## 首次配置

1. **配置 API Key**
```bash
# 将API密钥添加到环境变量
echo 'export DASHSCOPE_API_KEY="您的API密钥"' >> ~/.bashrc
source ~/.bashrc
```

2. **创建虚拟环境**（仅首次需要）
```bash
cd HAMD
python3 -m venv venv
```

## 快速启动步骤

1. **进入项目目录**
```bash
cd HAMD
```

2. **激活虚拟环境**
```bash
source venv/bin/activate  # Linux
# 或
.\venv\Scripts\activate  # Windows
```

3. **运行项目**
```bash
python src/app.py
```

4. **访问应用**
- 本地访问：http://127.0.0.1:5000

## 常见问题

1. **如果提示缺少依赖**
```bash
pip install flask flask-socketio eventlet openai dashscope
```

2. **如果需要退出程序**
- 按 `Ctrl+C` 停止服务器
- 输入 `deactivate` 退出虚拟环境

## 注意事项
- 确保 API Key 已正确设置
- 确保在运行程序前已激活虚拟环境（命令行前缀显示 `(venv)`）
- 首次运行可能需要等待几秒钟加载模型

## 文件说明
- `src/app.py`: 主程序入口
- `src/templates/`: 前端页面
- `newprompt.txt`: 提示词配置
- `assessment_results/`: 评估结果保存目录
- `progress/`: 进度保存目录