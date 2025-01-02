import os
import sys
import warnings
from functools import wraps

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from flask import Flask, render_template, request, Response
from flask_socketio import emit
import asyncio
from src.core.assessment_framework import AssessmentFramework
from src.utils.globals import socketio, init_socketio
from src.utils.prompt_parser import PromptParser

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# 添加认证配置
def check_auth(username, password):
    return username == "admin" and password == os.getenv('ACCESS_CODE', 'testcode')

def authenticate():
    return Response(
        '请登录', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

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

# 初始化评估框架
framework = AssessmentFramework(prompt_file_path, model_config)
framework.initialize_items_from_prompts()

# 删除本地的get_question函数，使用PromptParser中的函数
get_question = PromptParser.get_question

# 在主页路由上添加认证装饰器
@app.route('/')
@requires_auth
def index():
    return render_template('index.html')

init_socketio(app)

@socketio.on('connect')
def handle_connect():
    pass

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
        current_item = framework.items[framework.current_item_index]
        user_response = data['content']
        
        # 获取当前条目的问题
        question = get_question(current_item.prompt)
        
        # 记录用户的回答
        framework.conversation_history[current_item.item_id].append({
            'user': user_response
        })
        
        # 获取当前条目的历史对话
        history = framework.conversation_history[current_item.item_id][:-1]
        
        # 将问题作为参数传递给 process_response
        result = await framework.process_response(user_response, history, question)
        
        if result['type'] == 'score':
            if not framework.save_progress():
                print("警告：评分后保存进度失败")
            
            next_item = framework.next_item()
            if next_item:
                print(f"切换到下一题: {framework.current_item_index + 1}")
                if not framework.save_progress():
                    print("警告：切换题目后保存进度失败")
                
                socketio.emit('message', {
                    'type': 'status',
                    'current_item': f"第 {framework.current_item_index + 1} 题",
                    'current_index': framework.current_item_index,
                    'total_items': len(framework.items)
                })
                
                question = get_question(next_item.prompt)
                framework.conversation_history[next_item.item_id].append({
                    'assistant': question
                })
                
                socketio.emit('message', {
                    'type': 'message',
                    'role': 'system',
                    'content': question
                })
            else:
                print("所有条目评估完成")
                framework.save_assessment_result()
                
                socketio.emit('message', {
                    'type': 'assessment_complete',
                    'content': '评估已完成，感谢您的参与！'
                })
                
                print(f"患者 {framework.patient_info['id']} 的评估已完成")
                print(f"总评分项目数: {len(framework.scores)}")
                
        else:
            if result.get('show_response', True):
                socketio.emit('message', {
                    'type': 'message',
                    'role': 'assistant',
                    'content': result['data']
                })
                
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
                question = get_question(current_item.prompt)
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
            
            question = get_question(current_item.prompt)
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
    port = int(os.getenv('PORT', 7860))
    host = os.getenv('HOST', '0.0.0.0')
    socketio.run(app, 
                host=host,
                port=port,
                debug=False,
                allow_unsafe_werkzeug=True)