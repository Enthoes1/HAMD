# 精神科量表评估系统

基于大语言模型的精神科量表评估系统，通过实时对话方式进行自动化评估。

## 主要功能

- 基于大语言模型的智能问答
- 实时WebSocket通信
- 自动评分与进度保存  
- 支持中断恢复
- 语音交互（开发中）

## 项目结构

```
├── src/                    # 源代码
│   ├── app.py             # 主程序入口
│   ├── core/              # 核心模块
│   ├── llm/               # LLM处理
│   ├── utils/             # 工具类
│   ├── speech/            # 语音模块
│   └── templates/         # 前端页面
└── assessment_results/    # 评估结果
```

## 快速开始

1. 安装依赖
bash
pip install -r requirements.txt
```

2. 配置环境变量
```bash
export DASHSCOPE_API_KEY=your_api_key
```

3. 运行程序
```bash
python src/app.py
```

4. 访问系统
- 打开浏览器访问 http://localhost:5000
- 输入患者基本信息开始评估
- 查看实时评估状态

## 技术特点

- **评估框架**：管理评估流程、状态和数据存储
- **LLM集成**：对接通义千问API，智能解析对话
- **数据管理**：自动保存评估结果和对话历史
- **实时通信**：基于WebSocket的即时响应

## 评估结果

系统自动生成JSON格式的评估结果：
```json
{
  "timestamp": "20240101_123456",
  "patient_info": {
    "id": "P001",
    "name": "张三"
  },
  "scores": {
    "1": {"score": 2},
    "2": {"score": 1}
  }
}
```

## 开发计划

- 语音交互优化
- 评估报告生成
- 用户界面改进
- 多模型支持

## 问题反馈

如有问题或建议，请提交Issue或Pull Request。

## 许可证

MIT License
```