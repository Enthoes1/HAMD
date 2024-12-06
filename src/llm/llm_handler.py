import os
from openai import OpenAI
import json
from utils.globals import socketio

class LLMHandler:
    def __init__(self, model_config):
        self.client = OpenAI(
            api_key=model_config.get('api_key', os.getenv("DASHSCOPE_API_KEY")),
            base_url=model_config.get('base_url', "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self.model = model_config.get('model', 'qwen-plus')
        self.parameters = model_config.get('parameters', {})
        
    async def evaluate_response(self, prompt, user_response, conversation_history=None, question=None):
        try:
            # 构建消息列表
            messages = [
                {'role': 'system', 'content': prompt},
            ]
            
            # 添加问题作为第一条 assistant 消息
            if question:
                messages.append({'role': 'assistant', 'content': question})
            
            # 添加历史对话（不包括当前用户输入）
            if conversation_history:
                for entry in conversation_history:
                    if entry.get('user'):
                        messages.append({'role': 'user', 'content': entry['user']})
                    if entry.get('assistant'):
                        messages.append({'role': 'assistant', 'content': entry['assistant']})
            
            # 添加当前用户输入
            messages.append({'role': 'user', 'content': user_response})
            
            # 发送消息列表到前端显示
            if socketio:
                socketio.emit('message', {
                    'type': 'llm_messages',
                    'messages': messages
                })
            
            # 调用LLM进行评估
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.parameters.get('temperature', 0.7),
                top_p=self.parameters.get('top_p', 0.8),
                max_tokens=self.parameters.get('max_tokens', 1500),
                presence_penalty=self.parameters.get('presence_penalty', 0),
                frequency_penalty=self.parameters.get('frequency_penalty', 0),
                repetition_penalty=self.parameters.get('repetition_penalty', 1.1),
                stop=self.parameters.get('stop', None)
            )
            
            # 获取响应文本
            response = completion.choices[0].message.content
            
            # 尝试解析JSON评分
            score = self._try_parse_score(response)
            if score:
                # 返回评分结果，但不包含原始响应
                return {'type': 'score', 'data': score, 'show_response': False}
            else:
                # 检查响应中是否包含分数相关内容
                score_keywords = ['0分', '1分', '2分', '3分', '4分']
                if any(keyword in response for keyword in score_keywords):
                    # 如果包含分数相关内容，重新发送请求
                    messages.append({
                        'role': 'system', 
                        'content': "请不要与患者讨论分数等内容，如果根据现有对话能够判断分数，则直接输出json；如果不能，请继续追问。"
                    })
                    
                    # 再次调用LLM
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.parameters.get('temperature', 0.7),
                        top_p=self.parameters.get('top_p', 0.8),
                        max_tokens=self.parameters.get('max_tokens', 1500),
                        presence_penalty=self.parameters.get('presence_penalty', 0),
                        frequency_penalty=self.parameters.get('frequency_penalty', 0),
                        repetition_penalty=self.parameters.get('repetition_penalty', 1.1),
                        stop=self.parameters.get('stop', None)
                    )
                    
                    # 获取新的响应
                    new_response = completion.choices[0].message.content
                    
                    # 再次尝试解析JSON
                    score = self._try_parse_score(new_response)
                    if score:
                        return {'type': 'score', 'data': score, 'show_response': False}
                    else:
                        return {'type': 'message', 'data': new_response, 'show_response': True}
                else:
                    # 如果不包含分数相关内容，返回原始响应
                    return {'type': 'message', 'data': response, 'show_response': True}
            
        except Exception as e:
            print(f"LLM调用出错: {str(e)}")
            raise
            
    def _try_parse_score(self, response):
        """尝试从响应中解析评分JSON"""
        try:
            # 找到所有可能的JSON对象
            json_objects = []
            start = 0
            while True:
                start = response.find('{', start)
                if start == -1:
                    break
                    
                # 查找匹配的结束括号
                stack = []
                end = start
                for i in range(start, len(response)):
                    if response[i] == '{':
                        stack.append('{')
                    elif response[i] == '}':
                        stack.pop()
                        if not stack:  # 找到匹配的结束括号
                            end = i + 1
                            break
                
                if end > start:
                    try:
                        json_str = response[start:end]
                        score_data = json.loads(json_str)
                        # 验证是否是评分数据
                        if 'score' in score_data or ('label' in score_data and 'score' in score_data):
                            # 如果是字典形式的score，拆分成多个评分对象
                            if isinstance(score_data.get('score', {}), dict):
                                for label, score in score_data['score'].items():
                                    json_objects.append({
                                        'label': int(label),
                                        'score': score
                                    })
                            else:
                                json_objects.append(score_data)
                    except:
                        pass
                    
                start = end + 1
                
            # 如果找到有效的评分JSON对象
            if json_objects:
                # 确保每个对象都有正确的格式
                formatted_objects = []
                for obj in json_objects:
                    if 'score' in obj:
                        formatted_obj = {
                            'label': obj.get('label', 1),  # 使用默认标签1如果没有指定
                            'score': obj['score']
                        }
                        formatted_objects.append(formatted_obj)
                
                # 如果只有一个评分对象，直接返回
                if len(formatted_objects) == 1:
                    return formatted_objects[0]
                # 如果有多个评分对象，返回数组
                else:
                    return formatted_objects
                
            return None
        except Exception as e:
            print(f"JSON解析错误: {str(e)}")
            return None