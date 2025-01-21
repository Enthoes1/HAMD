import os
import sys
import warnings
from functools import wraps
from datetime import datetime
import json

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_socketio import emit
import asyncio
from core.assessment_framework import AssessmentFramework
from utils.globals import socketio, init_socketio
from utils.prompt_parser import PromptParser

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hamd2024_secure_key_!@#$%^&*()'
init_socketio(app)

# 配置访问密码
ACCESS_CODE = "hamd2024"  # 你可以修改这个密码

def check_auth():
    """检查用户是否已认证"""
    return session.get('authenticated', False)

# 配置模型参数
model_config = {
    'api_key': os.getenv("DASHSCOPE_API_KEY"),
    'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
    'model': 'qwen-plus',
    'parameters': {
        'temperature': 0.7,      # 温度参数，控制输出的随机性，范围 0-1
        'top_p': 0.6,           # 控制输出的多样性，范围 0-1
        'max_tokens': 1500,     # 最大输出 token 数
    }
}

# 获取项目根目录路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
prompt_file_path = os.path.join(root_dir, "newprompt.txt")

# 用户评估框架字典
user_frameworks = {}

def get_framework(sid):
    """获取或创建用户的评估框架"""
    if sid not in user_frameworks:
        framework = AssessmentFramework(prompt_file_path, model_config)
        framework.initialize_items_from_prompts()
        user_frameworks[sid] = framework
    return user_frameworks[sid]

@app.route('/')
def index():
    if not check_auth():
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/phq9')
def phq9():
    if not check_auth():
        return redirect(url_for('login'))
    return render_template('phq9.html')

@app.route('/save_phq9', methods=['POST'])
def save_phq9():
    if not check_auth():
        return jsonify({'error': '未授权访问'}), 401
        
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        
        if not patient_id:
            return jsonify({'error': '缺少患者ID'}), 400
            
        # 创建PHQ9结果保存目录
        phq9_dir = os.path.join(os.path.dirname(prompt_file_path), "phq9_results")
        if not os.path.exists(phq9_dir):
            os.makedirs(phq9_dir)
            
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"phq9_{patient_id}_{timestamp}.json"
        filepath = os.path.join(phq9_dir, filename)
        
        # 保存结果
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return jsonify({'success': True, 'message': '评估结果已保存'})
        
    except Exception as e:
        print(f"保存PHQ9结果时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['GET', 'POST'])
def login():
    # 如果是GET请求，清除session，确保重新登录
    if request.method == 'GET':
        session.clear()
        return render_template('login.html')
        
    if request.method == 'POST':
        if request.form.get('password') == ACCESS_CODE:
            session['authenticated'] = True
            return 'success'
        return 'error'

@socketio.on('connect')
def handle_connect():
    if not check_auth():
        return False  # 拒绝未认证的WebSocket连接
    return True

@socketio.on('disconnect')
def handle_disconnect():
    """清理用户的评估框架"""
    sid = request.sid
    if sid in user_frameworks:
        # 保存最终结果
        framework = user_frameworks[sid]
        if framework.patient_info:
            framework.save_assessment_result()
        del user_frameworks[sid]

@socketio.on('user_input')
def handle_message(data):
    try:
        framework = get_framework(request.sid)
        current_item = framework.items[framework.current_item_index]
        user_response = data['content']
        sid = request.sid  # 保存当前的session id
        
        # 获取当前条目的问题
        question = PromptParser.get_question(current_item.prompt)
        
        # 记录用户的回答
        framework.conversation_history[current_item.item_id].append({
            'user': user_response
        })
        
        # 获取当前条目的历史对话
        history = framework.conversation_history[current_item.item_id][:-1]
        
        async def process():
            # 将问题作为参数传递给 process_response
            result = await framework.process_response(user_response, history, question)
            return result
            
        def async_process():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(process())
            loop.close()
            
            if result['type'] == 'score':
                if not framework.save_progress():
                    print("警告：评分后保存进度失败")
                
                next_item = framework.next_item()
                if next_item:
                    print(f"切换到下一题: {framework.current_item_index + 1}")
                    if not framework.save_progress():
                        print("警告：切换题目后保存进度失败")
                    
                    question = PromptParser.get_question(next_item.prompt)
                    framework.conversation_history[next_item.item_id].append({
                        'assistant': question
                    })
                    
                    socketio.emit('message', {
                        'type': 'message',
                        'role': 'system',
                        'content': question
                    }, room=sid)  # 使用保存的sid
                else:
                    print("所有条目评估完成")
                    framework.save_assessment_result()
                    
                    socketio.emit('message', {
                        'type': 'assessment_complete',
                        'content': '评估已完成，感谢您的参与！您现在可以点击右上角的"PHQ-9自评"按钮进行自评。'
                    }, room=sid)  # 使用保存的sid
                    
                    print(f"患者 {framework.patient_info['id']} 的评估已完成")
                    print(f"总评分项目数: {len(framework.scores)}")
                    
            else:
                if result.get('show_response', True):
                    socketio.emit('message', {
                        'type': 'message',
                        'role': 'assistant',
                        'content': result['data']
                    }, room=sid)  # 使用保存的sid
        
        socketio.start_background_task(async_process)
        
    except Exception as e:
        print(f"消息处理错误: {str(e)}")
        emit('message', {
            'type': 'message',
            'role': 'system',
            'content': f"错误：{str(e)}"
        })

@socketio.on('submit_patient_info')
def handle_patient_info(data):
    try:
        framework = get_framework(request.sid)
        
        if 'id' not in data:
            print("错误：患者信息中缺少ID")
            emit('message', {
                'type': 'error',
                'content': '患者信息不完整，请确保包含ID'
            })
            return
            
        if framework.load_progress(data['id']):
            print(f"已恢复用户 {data['id']} 的进度")
            current_item = framework.items[framework.current_item_index]
            
            emit('message', {
                'type': 'status',
                'current_item': f"第 {framework.current_item_index + 1} 题",
                'current_index': framework.current_item_index,
                'total_items': len(framework.items)
            })
            
            history = framework.conversation_history[current_item.item_id]
            for i, entry in enumerate(history):
                if 'assistant' in entry:
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
            
            if not history:
                question = PromptParser.get_question(current_item.prompt)
                emit('message', {
                    'type': 'message',
                    'role': 'system',
                    'content': question
                })
        else:
            framework.set_patient_info(data)
            print(f"新用户 {data['id']} 开始评估")
            if not framework.save_progress():
                print("警告：初始状态保存失败")
            
            current_item = framework.items[framework.current_item_index]
            emit('message', {
                'type': 'status',
                'current_item': f"第 {framework.current_item_index + 1} 题",
                'current_index': framework.current_item_index,
                'total_items': len(framework.items)
            })
            
            question = PromptParser.get_question(current_item.prompt)
            emit('message', {
                'type': 'message',
                'role': 'system',
                'content': question
            })
            
    except Exception as e:
        print(f"处理患者信息错误: {str(e)}")
        emit('message', {
            'type': 'message',
            'role': 'system',
            'content': f"错误：{str(e)}"
        })

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)