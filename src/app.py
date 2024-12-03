import os
import sys
import warnings

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template
from flask_socketio import emit
import asyncio
from core.assessment_framework import AssessmentFramework
from utils.globals import socketio, init_socketio
from speech.speech_handler import SpeechHandler  # 保留语音处理器导入
from speech.speech_recognition import SpeechRecognition

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
init_socketio(app)

# 配置模型参数
model_config = {
    'api_key': os.getenv("DASHSCOPE_API_KEY"),
    'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
    'model': 'qwen-plus'
}

# 获取项目根目录路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prompt_file_path = os.path.join(root_dir, "newprompt.txt")#在此处修改提示词指向

# 初始化评估框架
framework = AssessmentFramework(prompt_file_path, model_config)
framework.initialize_items_from_prompts()

# 初始化语音处理器（仅用于TTS）
speech_handler = SpeechHandler()

# 初始化语音识别器
speech_recognizer = SpeechRecognition()

def get_question(prompt):
    """从提示词中提取问诊问题"""
    try:
        # 处理嵌套的JSON字符串问题
        # 将内部的JSON格式字符串中的引号替换为单引号
        import re
        processed_prompt = re.sub(r'({"label":[^}]+})', lambda m: m.group(1).replace('"', "'"), prompt)
        
        # 将提示词解析为JSON
        import json
        prompt_data = json.loads(processed_prompt)
        
        # 从条目详情中提取问题
        if "条目详情" in prompt_data and "问题" in prompt_data["条目详情"]:
            return prompt_data["条目详情"]["问题"]
            
        return "请描述您的情况。"
    except Exception as e:
        print(f"提取问题出错: {str(e)}")
        print(f"原始提示词: {prompt}")
        # 如果JSON解析失败，尝试使用正则表达式直接提取
        try:
            match = re.search(r'"问题":\s*"([^"]+)"', prompt)
            if match:
                return match.group(1)
        except Exception as e2:
            print(f"正则提取失败: {str(e2)}")
        return "请描述您的情况。"

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    try:
        # 连接时不做任何操作，等待用户填写基本信息
        pass
    except Exception as e:
        print(f"连接处理错误: {str(e)}")
        emit('message', {
            'type': 'message',
            'role': 'system',
            'content': f"初始化错误：{str(e)}"
        })

@socketio.on('user_input')
def handle_message(data):
    try:
        def async_process():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(process_message(data))
            loop.close()
        
        socketio.start_background_task(async_process)
    except Exception as e:
        print(f"消息处理错误: {str(e)}")
        emit('message', {
            'type': 'message',
            'role': 'system',
            'content': f"错误：{str(e)}"
        })

async def process_message(data):
    try:
        result = await framework.process_response(data['content'])
        
        if result['type'] == 'score':
            # 如果评估完成，切换到下一个条目
            next_item = framework.next_item()
            if next_item:
                # 更新状态
                socketio.emit('message', {
                    'type': 'status',
                    'current_item': f"第 {framework.current_item_index + 1} 题",
                    'current_index': framework.current_item_index,
                    'total_items': len(framework.items)
                })
                
                # 提取并发送下一个问诊问题
                question = get_question(next_item.prompt)
                # 先发送文本
                socketio.emit('message', {
                    'type': 'message',
                    'role': 'system',
                    'content': question
                })
                
                # 异步播放语音
                def async_tts():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(speech_handler.text_to_speech(question))
                    loop.close()
                
                socketio.start_background_task(async_tts)
                
            else:
                # 评估完成
                completion_message = "评估完成！"
                # 先发送文本
                socketio.emit('message', {
                    'type': 'message',
                    'role': 'system',
                    'content': completion_message
                })
                
                # 异步播放语音
                def async_tts():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(speech_handler.text_to_speech(completion_message))
                    loop.close()
                
                socketio.start_background_task(async_tts)
                
        else:
            if result.get('show_response', True):
                # 先发送文字到前端
                socketio.emit('message', {
                    'type': 'message',
                    'role': 'assistant',
                    'content': result['data']
                })
                
                # 异步播放语音
                def async_tts():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(speech_handler.text_to_speech(result['data']))
                    loop.close()
                
                socketio.start_background_task(async_tts)
                
    except Exception as e:
        print(f"异步处理错误: {str(e)}")
        socketio.emit('message', {
            'type': 'message',
            'role': 'system',
            'content': f"错误：{str(e)}"
        })

@socketio.on('submit_patient_info')
def handle_patient_info(data):
    try:
        # 保存患者信息
        framework.set_patient_info(data)
        
        # 发送初始状态
        current_item = framework.items[framework.current_item_index]
        emit('message', {
            'type': 'status',
            'current_item': f"第 {framework.current_item_index + 1} 题",
            'current_index': framework.current_item_index,
            'total_items': len(framework.items)
        })
        
        # 提取并发送第一个问诊问题
        question = get_question(current_item.prompt)
        emit('message', {
            'type': 'message',
            'role': 'system',
            'content': question
        })
        
        # 播放问题语音
        def async_tts():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(speech_handler.text_to_speech(question))
            loop.close()
        
        socketio.start_background_task(async_tts)
        
    except Exception as e:
        print(f"处理患者信息错误: {str(e)}")
        emit('message', {
            'type': 'message',
            'role': 'system',
            'content': f"错误：{str(e)}"
        })

@socketio.on('start_recording')
def handle_start_recording():
    try:
        speech_recognizer.start_recording()
    except Exception as e:
        print(f"开始录音错误: {str(e)}")
        emit('message', {
            'type': 'message',
            'role': 'system',
            'content': f"录音错误：{str(e)}"
        })

@socketio.on('stop_recording')
def handle_stop_recording():
    try:
        # 停止录音并获取识别结果
        text = speech_recognizer.stop_recording()
        
        # 发送识别结果到前端
        emit('transcription', {
            'text': text
        })
    except Exception as e:
        print(f"停止录音错误: {str(e)}")
        emit('message', {
            'type': 'message',
            'role': 'system',
            'content': f"录音错误：{str(e)}"
        })

if __name__ == '__main__':
    socketio.run(app, debug=True) 