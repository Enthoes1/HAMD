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
ACCESS_CODE = "hamd2024"  # 普通用户密码
ADMIN_CODE = "hamd2024_admin"  # 管理员密码

def check_auth():
    """检查用户是否已认证"""
    return session.get('authenticated', False)

def check_admin():
    """检查用户是否为管理员"""
    return session.get('is_admin', False)

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
    """显示主页面"""
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
        password = request.form.get('password')
        if password == ADMIN_CODE:
            session['authenticated'] = True
            session['is_admin'] = True
            return jsonify({
                'status': 'success',
                'is_admin': True
            })
        elif password == ACCESS_CODE:
            session['authenticated'] = True
            session['is_admin'] = False
            return jsonify({
                'status': 'success',
                'is_admin': False
            })
        return jsonify({
            'status': 'error',
            'message': '密码错误'
        })

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

@app.route('/report')
def report():
    """显示评估报告页面"""
    return render_template('report.html')

@app.route('/get_report')
def get_report():
    """获取评估报告数据"""
    try:
        patient_id = request.args.get('patient_id')
        if not patient_id:
            return jsonify({'error': '未提供患者ID'}), 400
            
        # 获取评估结果目录
        assessment_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assessment_results")
        phq9_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phq9_results")
        
        # 查找最新的HAMD评估
        hamd_files = [f for f in os.listdir(assessment_dir) if f.startswith(f'hamd_{patient_id}_')]
        if not hamd_files:
            return jsonify({'error': '未找到HAMD评估记录'}), 404
            
        # 按时间戳排序，获取最新的评估
        latest_hamd = max(hamd_files, key=lambda x: os.path.getctime(os.path.join(assessment_dir, x)))
        timestamp = latest_hamd.split('_')[-1].split('.')[0]
        
        # 读取HAMD结果
        hamd_path = os.path.join(assessment_dir, latest_hamd)
        with open(hamd_path, 'r', encoding='utf-8') as f:
            hamd_data = json.load(f)
            
        # 查找对应的PHQ-9结果
        phq9_file = f"phq9_{patient_id}_{timestamp}.json"
        phq9_path = os.path.join(phq9_dir, phq9_file)
        phq9_data = None
        
        if os.path.exists(phq9_path):
            with open(phq9_path, 'r', encoding='utf-8') as f:
                phq9_data = json.load(f)
        else:
            # 如果没找到对应时间戳的PHQ-9，尝试找最新的PHQ-9
            phq9_files = [f for f in os.listdir(phq9_dir) if f.startswith(f'phq9_{patient_id}_')]
            if phq9_files:
                latest_phq9 = max(phq9_files, key=lambda x: os.path.getctime(os.path.join(phq9_dir, x)))
                with open(os.path.join(phq9_dir, latest_phq9), 'r', encoding='utf-8') as f:
                    phq9_data = json.load(f)
        
        # 准备报告数据
        report_data = {
            'patient_info': hamd_data['patient_info'],
            'hamd': {
                'total_score': hamd_data['total_score'],
                'severity': get_hamd_severity(hamd_data['total_score']),
                'scores': hamd_data['scores']
            }
        }
        
        if phq9_data:
            report_data['phq9'] = {
                'total_score': phq9_data['total_score'],
                'interpretation': phq9_data['interpretation'],
                'answers': phq9_data.get('answers', [])  # 使用get方法避免键不存在的错误
            }
        
        return jsonify(report_data)
        
    except Exception as e:
        print(f"获取报告数据失败: {str(e)}")
        return jsonify({'error': str(e)}), 500

def get_hamd_severity(total_score):
    """根据HAMD总分判断严重程度"""
    if total_score < 7:
        return "无抑郁"
    elif total_score < 17:
        return "轻度抑郁"
    elif total_score < 24:
        return "中度抑郁"
    else:
        return "重度抑郁"

@app.route('/admin')
def admin():
    """显示管理页面"""
    if not check_auth() or not check_admin():
        return redirect(url_for('login'))
    return render_template('admin.html')

@app.route('/get_all_patients')
def get_all_patients():
    """获取所有患者及其评估记录"""
    if not check_auth() or not check_admin():
        return jsonify({'error': '未授权访问'}), 401
        
    try:
        patients = {}
        
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        progress_dir = os.path.join(root_dir, "progress")
        results_dir = os.path.join(root_dir, "assessment_results")
        
        # 确保目录存在
        os.makedirs(progress_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # 读取进行中的评估
        if os.path.exists(progress_dir):
            for file in os.listdir(progress_dir):
                if file.startswith('progress_'):
                    patient_id = file.replace('progress_', '').replace('.json', '')
                    with open(os.path.join(progress_dir, file), 'r', encoding='utf-8') as f:
                        progress_data = json.load(f)
                        patient_info = progress_data.get('patient_info', {})
                        
                        if patient_id not in patients:
                            patients[patient_id] = {
                                'id': patient_id,
                                'name': patient_info.get('name', '未知'),
                                'gender': patient_info.get('gender', '未知'),
                                'age': patient_info.get('age', '未知'),
                                'assessments': []
                            }
                            
                        # 添加进行中的评估
                        assessment_info = {
                            'id': 'current',
                            'timestamp': datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
                            'completed': False,
                            'current_item': progress_data.get('current_item_index', 0) + 1
                        }
                        
                        # 检查是否已存在相同的评估记录
                        if not any(a['id'] == 'current' for a in patients[patient_id]['assessments']):
                            patients[patient_id]['assessments'].append(assessment_info)
        
        # 读取已完成的评估
        if os.path.exists(results_dir):
            for file in os.listdir(results_dir):
                if file.startswith('hamd_'):
                    parts = file.replace('.json', '').split('_')
                    patient_id = parts[1]
                    timestamp = '_'.join(parts[2:])
                    
                    with open(os.path.join(results_dir, file), 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                        patient_info = result_data.get('patient_info', {})
                        
                        if patient_id not in patients:
                            patients[patient_id] = {
                                'id': patient_id,
                                'name': patient_info.get('name', '未知'),
                                'gender': patient_info.get('gender', '未知'),
                                'age': patient_info.get('age', '未知'),
                                'assessments': []
                            }
                        else:
                            # 更新患者信息（使用最新的信息）
                            patients[patient_id].update({
                                'name': patient_info.get('name', patients[patient_id]['name']),
                                'gender': patient_info.get('gender', patients[patient_id]['gender']),
                                'age': patient_info.get('age', patients[patient_id]['age'])
                            })
                            
                        # 添加已完成的评估
                        assessment_info = {
                            'id': timestamp,
                            'timestamp': datetime.strptime(timestamp, "%Y%m%d_%H%M%S").strftime("%Y/%m/%d %H:%M:%S"),
                            'completed': True,
                            'total_score': result_data.get('total_score', 0)
                        }
                        
                        # 检查是否已存在相同的评估记录
                        if not any(a['id'] == timestamp for a in patients[patient_id]['assessments']):
                            patients[patient_id]['assessments'].append(assessment_info)
        
        # 将字典转换为列表并按ID排序
        patient_list = list(patients.values())
        patient_list.sort(key=lambda x: x['id'])
        
        return jsonify(patient_list)
        
    except Exception as e:
        print(f"获取患者列表时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_patient_info')
def get_patient_info():
    """获取指定患者的信息"""
    try:
        patient_id = request.args.get('patient_id')
        if not patient_id:
            return jsonify({'error': '缺少患者ID'}), 400
            
        # 首先尝试从进行中的评估获取信息
        progress_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                   "progress", f"progress_{patient_id}.json")
        
        if os.path.exists(progress_file):
            with open(progress_file, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                return jsonify(progress_data['patient_info'])
                
        # 如果没有进行中的评估，尝试从已完成的评估获取信息
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "assessment_results")
        
        if os.path.exists(results_dir):
            result_files = [f for f in os.listdir(results_dir) 
                          if f.startswith(f'hamd_{patient_id}_')]
            if result_files:
                # 使用最新的评估结果
                latest_file = max(result_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
                with open(os.path.join(results_dir, latest_file), 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                    return jsonify(result_data['patient_info'])
        
        return jsonify({'error': '未找到患者信息'}), 404
        
    except Exception as e:
        print(f"获取患者信息时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete_assessment', methods=['POST'])
def delete_assessment():
    """删除评估记录"""
    if not check_auth() or not check_admin():
        return jsonify({'error': '未授权访问'}), 401
        
    try:
        data = request.get_json()
        patient_id = data.get('patient_id')
        assessment_id = data.get('assessment_id')
        
        if not patient_id or not assessment_id:
            return jsonify({'error': '缺少必要参数'}), 400
            
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 如果是进行中的评估
        if assessment_id == 'current':
            progress_file = os.path.join(root_dir, "progress", f"progress_{patient_id}.json")
            if os.path.exists(progress_file):
                os.remove(progress_file)
                print(f"已删除进行中的评估: {progress_file}")
        else:
            # 如果是已完成的评估
            results_dir = os.path.join(root_dir, "assessment_results")
            result_file = os.path.join(results_dir, f"hamd_{patient_id}_{assessment_id}.json")
            if os.path.exists(result_file):
                os.remove(result_file)
                print(f"已删除已完成的评估: {result_file}")
                
            # 同时删除对应的PHQ9评估（如果存在）
            phq9_dir = os.path.join(root_dir, "phq9_results")
            phq9_file = os.path.join(phq9_dir, f"phq9_{patient_id}_{assessment_id}.json")
            if os.path.exists(phq9_file):
                os.remove(phq9_file)
                print(f"已删除对应的PHQ9评估: {phq9_file}")
        
        return jsonify({'success': True, 'message': '评估记录已删除'})
        
    except Exception as e:
        print(f"删除评估记录时出错: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)