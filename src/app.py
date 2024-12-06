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

# 添加一个全局的语音播放标志
speech_playing = False

async def play_speech(text):
    """异步播放语音"""
    global speech_playing
    try:
        speech_playing = True
        print(f"开始播放语音: {text[:30]}...")  # 添加日志
        await speech_handler.text_to_speech(text)
        print("语音播放完成")
    except Exception as e:
        print(f"语音播放错误: {str(e)}")
    finally:
        speech_playing = False

def async_play_speech(text):
    """创建新的事件循环来播放语音"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(play_speech(text))
        loop.close()
    except Exception as e:
        print(f"异步语音播放错误: {str(e)}")

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
        # 如果正播放语音，先停止播放
        global speech_playing
        if speech_playing:
            # 这里可以添加停止语音播放的逻辑
            speech_playing = False
            
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
        current_item = framework.items[framework.current_item_index]
        user_response = data['content']
        
        # 获取当前条目的问题
        question = get_question(current_item.prompt)
        
        # 记录用户的回答
        framework.conversation_history[current_item.item_id].append({
            'user': user_response
        })
        
        # 获取当前条目的历史对话（不包括刚刚添加的用户回答）
        history = framework.conversation_history[current_item.item_id][:-1]
        
        # 将问题作为参数传递给 process_response
        result = await framework.process_response(user_response, history, question)
        
        if result['type'] == 'score':
            # 保存评分后的进度
            if not framework.save_progress():
                print("警告：评分后保存进度失败")
            
            next_item = framework.next_item()
            if next_item:
                # 更新状态并保存新进度
                print(f"切换到下一题: {framework.current_item_index + 1}")
                if not framework.save_progress():  # 保存切换到新条目后的进度
                    print("警告：切换题目后保存进度失败")
                
                socketio.emit('message', {
                    'type': 'status',
                    'current_item': f"第 {framework.current_item_index + 1} 题",
                    'current_index': framework.current_item_index,
                    'total_items': len(framework.items)
                })
                
                # 提取并发送下一个问诊问题
                question = get_question(next_item.prompt)
                
                # 将问题添加到对话历史
                framework.conversation_history[next_item.item_id].append({
                    'assistant': question
                })
                
                # 先发送文本
                socketio.emit('message', {
                    'type': 'message',
                    'role': 'system',
                    'content': question
                })
                
                # 异步播放语音
                socketio.start_background_task(async_play_speech, question)
            else:
                # 最后一个条目评估完成
                print("所有条目评估完成")
                
                # 保存最终评估结果
                framework.save_assessment_result()
                
                # 发送完成消息到前端
                socketio.emit('message', {
                    'type': 'assessment_complete',
                    'content': '评估已完成，感谢您的参与！您的评估结果已保存。'
                })
                
                # 可以在这里添加其他完成后的处理逻辑
                print(f"患者 {framework.patient_info['id']} 的评估已完成")
                print(f"总评分项目数: {len(framework.scores)}")
                
        else:
            if result.get('show_response', True):
                # 不再重复添加到历史记录，因为 framework 已经添加过了
                # framework.conversation_history[current_item.item_id].append({
                #     'assistant': result['data']
                # })
                
                # 只发送到前端显示
                socketio.emit('message', {
                    'type': 'message',
                    'role': 'assistant',
                    'content': result['data']
                })
                
                # 异步播放语音
                socketio.start_background_task(async_play_speech, result['data'])
                
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
        # 检查患者ID
        if 'id' not in data:
            print("错误：患者信息中缺少ID")
            emit('message', {
                'type': 'error',
                'content': '患者信息不完整，请确保包含ID'
            })
            return
            
        # 检查是否有保存的进度
        if framework.load_progress(data['id']):
            print(f"已恢复用户 {data['id']} 的进度")
            current_item = framework.items[framework.current_item_index]
            
            # 发送当前状
            emit('message', {
                'type': 'status',
                'current_item': f"第 {framework.current_item_index + 1} 题",
                'current_index': framework.current_item_index,
                'total_items': len(framework.items)
            })
            
            # 发送当前条目的所有历史对话
            history = framework.conversation_history[current_item.item_id]
            for i, entry in enumerate(history):
                if 'assistant' in entry:
                    # 第一条 assistant 消息使用 system 角色，其他使用 assistant 角色
                    role = 'system' if i == 0 else 'assistant'
                    emit('message', {
                        'type': 'message',
                        'role': role,
                        'content': entry['assistant']
                    })
                if 'user' in entry:
                    emit('message', {
                        'type': 'message',
                        'role': 'user',
                        'content': entry['user']
                    })
            
            # 如果当前条目还没有对话历史，发送问题
            if not history:
                question = get_question(current_item.prompt)
                emit('message', {
                    'type': 'message',
                    'role': 'system',
                    'content': question
                })
                # 异步播放语音
                socketio.start_background_task(async_play_speech, question)
                
        else:
            # 如果没有进度，保存新的患者信息
            framework.set_patient_info(data)
            print(f"新用户 {data['id']} 开始评估")
            if not framework.save_progress():  # 保存初始状态
                print("警告：初始状态保存失败")
            
            # 发送初始状态
            current_item = framework.items[framework.current_item_index]
            emit('message', {
                'type': 'status',
                'current_item': f"第 {framework.current_item_index + 1} 题",
                'current_index': framework.current_item_index,
                'total_items': len(framework.items)
            })
            
            # 提取并发送第一个问题
            question = get_question(current_item.prompt)
            emit('message', {
                'type': 'message',
                'role': 'system',
                'content': question
            })
            
            # 异步播放语音
            socketio.start_background_task(async_play_speech, question)
            
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