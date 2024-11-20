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
        
    async def evaluate_response(self, prompt, user_response, conversation_history=None):
        try:
            # 构建消息列表
            messages = [
                # 系统角色：提供评估标准和任务说明
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
                return {'type': 'score', 'data': score}
            else:
                return {'type': 'message', 'data': response}
            
        except Exception as e:
            print(f"LLM调用出错: {str(e)}")
            raise
            
    def _try_parse_score(self, response):
        """尝试从响应中解析评分JSON"""
        try:
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end != -1:
                json_str = response[start:end]
                score_data = json.loads(json_str)
                # 验证是否是评分数据
                if 'score' in score_data or ('label' in score_data and any(k.endswith('score') for k in score_data.keys())):
                    return score_data
            return None
        except:
            return None