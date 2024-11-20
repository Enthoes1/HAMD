# 精神科量表评估系统

这是一个基于大语言模型的精神科量表评估系统，用于自动化进行精神科量表的评估过程。系统通过 WebSocket 实现实时对话，支持评分过程的自动保存，并提供清晰的评估进度展示。

## 项目结构

```
HAMD/
├── src/                      # 源代码目录
│   ├── app.py               # Flask应用主入口，处理WebSocket通信
│   ├── core/                # 核心功能模块
│   │   ├── __init__.py
│   │   └── assessment_framework.py  # 评估框架核心逻辑
│   ├── llm/                 # 大语言模型相关
│   │   ├── __init__.py
│   │   └── llm_handler.py   # LLM交互处理
│   ├── utils/               # 工具模块
│   │   ├── __init__.py
│   │   ├── globals.py       # 全局变量管理
│   │   └── prompt_parser.py # 提示词解析器
│   ├── templates/           # 前端模板
│   │   └── index.html       # 主页面
│   └── speech/              # 语音处理模块（待实现）
│       ├── __init__.py
│       └── speech_handler.py
├── assessment_results/      # 评估结果保存目录
│   └── assessment_*.json    # 评估结果文件
├── 提示词.txt               # 量表评估提示词
└── README.md               # 项目说明文档
```

## 核心功能模块

### 1. 评估框架 (AssessmentFramework)
- 管理评估流程和状态
- 处理用户响应
- 存储评分结果和对话历史
- 自动保存评估数据
- 控制条目切换

### 2. LLM处理器 (LLMHandler)
- 与大语言模型的交互
- 消息历史管理
- 评分结果解析
- JSON格式识别和处理

### 3. 提示词解析器 (PromptParser)
- 解析提示词文件
- 提取评估条目
- 维护提示词结构

### 4. WebSocket服务 (app.py)
- 处理实时通信
- 管理用户会话
- 提供Web界面
- 显示评估进度

## 数据流程

1. **初始化流程**
   - 加载提示词文件
   - 初始化评估条目
   - 建立WebSocket连接
   - 显示第一个问题

2. **评估流程**
   - 显示当前条目问题
   - 接收用户回答
   - 发送给LLM处理
   - 解析LLM响应
   - 存储评分/继续对话
   - 自动保存评估结果

3. **数据存储**
   - 评分数据实时保存
   - 对话历史记录
   - JSON格式结果文件

## 评估结果格式

评估结果保存为JSON文件，包含：
```json
{
  "timestamp": "20240101_123456",
  "scores": {
    "1": {"label": 1, "score": 2},
    "2": {"label": 2, "score": 1},
    "3": {"label": 3, "score": {"自杀": 1, "绝望感": 2}}
  },
  "conversation_history": {
    "1": [
      {"user": "用户输入", "assistant": "系统回复"}
    ]
  }
}
```

## 安装和运行

1. **安装依赖**
```bash
pip install flask flask-socketio eventlet openai
```

2. **环境变量设置**
```bash
export DASHSCOPE_API_KEY=your_api_key
```

3. **运行应用**
```bash
python src/app.py
```

4. **访问系统**
- 打开浏览器访问 http://localhost:5000
- 查看实时评估状态
- 右侧面板显示LLM交互详情

## 注意事项

1. **提示词格式**
   - 使用 #label# 标记不同条目
   - 包含完整的评估说明
   - 指定JSON格式的评分输出

2. **评分规则**
   - 自动识别JSON格式评分
   - 支持多级评分项目
   - 评分后自动保存

3. **对话管理**
   - 保存完整对话历史
   - 支持多轮交互
   - 显示实时状态

4. **数据存储**
   - 自动创建结果目录
   - 使用时间戳命名
   - 包含完整评估数据

## 开发计划

- [ ] 实现语音交互功能
- [ ] 添加评估报告生成
- [ ] 优化用户界面
- [ ] 添加数据分析功能