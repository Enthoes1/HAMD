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
        self.model = model_config.get('model', 'qwen-plus')#此处选择模型属性
        
    async def evaluate_response(self, prompt, user_response, conversation_history=None):
        try:
            # 构建消息列表
            messages = [
                {'role': 'system', 'content': prompt},
            ]
            
            # 添加历史对话
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
                messages=messages
            )
            
            # 获取响应文本
            response = completion.choices[0].message.content
            
            # 尝试解析JSON评分
            score = self._try_parse_score(response)
            if score:
                # 返回评分结果，但不包含原始响应
                return {'type': 'score', 'data': score, 'show_response': False}
            else:
                # 返回对话内容，需要显示
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
                # 查找下一个JSON开始位置
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
                        if 'score' in score_data and 'label' in score_data:
                            json_objects.append(score_data)
                    except:
                        pass
                    
                start = end + 1
                
            # 如果找到有效的评分JSON对象
            if json_objects:
                # 如果只有一个评分对象，直接返回
                if len(json_objects) == 1:
                    return json_objects[0]
                # 如果有多个评分对象，返回一个组合对象
                else:
                    combined_scores = {}
                    for obj in json_objects:
                        label = str(obj['label'])
                        if 'score' in obj:
                            if isinstance(obj['score'], dict):
                                combined_scores[label] = obj['score']
                            else:
                                combined_scores[label] = {'score': obj['score']}
                    return combined_scores
                
            return None
        except Exception as e:
            print(f"JSON解析错误: {str(e)}")
            return None