import os
from openai import OpenAI
import json
import dashscope
from src.utils.globals import socketio

class LLMHandler:
    def __init__(self, model_config):
        print(f"LLMHandler 初始化，使用模型: {model_config['model']}")
        print(f"API地址: {model_config['base_url']}")
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
                    if entry.get('role') == 'patient':
                        messages.append({'role': 'user', 'content': entry['content']})
                    elif entry.get('role') == 'assistant':
                        messages.append({'role': 'assistant', 'content': entry['content']})
            
            # 添加当前用户输入
            messages.append({'role': 'user', 'content': user_response})
            
            # 尝试发送消息列表到前端显示（如果socketio可用）
            try:
                if socketio:
                    socketio.emit('message', {
                        'type': 'llm_messages',
                        'messages': messages
                    })
            except:
                pass  # 如果socketio不可用，静默忽略
            
            # 调用LLM进行评估
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.parameters
            )
            
            # 获取响应文本
            response = completion.choices[0].message.content
            
            # 尝试解析JSON评分
            score = self._try_parse_score(response)
            if score:
                # 返回评分结果，同时保存原始响应
                return {
                    'type': 'score', 
                    'data': score, 
                    'raw_response': response,  # 添加原始响应
                    'show_response': False
                }
            else:
                # 检查响应中是否包含分数相关内容
                score_keywords = ['0分', '1分', '2分', '3分', '4分']
                if any(keyword in response for keyword in score_keywords):
                    # 将包含分数的响应加入到历史对话中
                    messages.append({
                        'role': 'assistant',
                        'content': response
                    })
                    
                    # 添加提示不要讨论分数的消息
                    messages.append({
                        'role': 'user', 
                        'content': "画外音：请不要与患者讨论分数等内容，如果根据现有对话能够判断分数，则直接输出json；如果不能，请继续追问。"
                    })
                    
                    # 再次调用LLM
                    completion = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        **self.parameters
                    )
                    
                    # 获取新的响应
                    new_response = completion.choices[0].message.content
                    
                    # 再次尝试解析JSON
                    score = self._try_parse_score(new_response)
                    if score:
                        return {
                            'type': 'score', 
                            'data': score, 
                            'raw_response': new_response,  # 添加原始响应
                            'show_response': False
                        }
                    else:
                        return {
                            'type': 'message', 
                            'data': new_response, 
                            'raw_response': new_response,  # 添加原始响应
                            'show_response': True
                        }
                else:
                    # 如果不包含分数相关内容，返回原始响应
                    return {
                        'type': 'message', 
                        'data': response, 
                        'raw_response': response,  # 添加原始响应
                        'show_response': True
                    }
                    
        except Exception as e:
            print(f"LLM调用出错: {str(e)}")
            raise
            
    def _try_parse_score(self, response):
        """尝试从响应中解析评分JSON"""
        try:
            # 找到所有可能的JSON对象
            json_objects = []
            current_pos = 0
            
            while current_pos < len(response):
                # 找到下一个左花括号
                start = response.find('{', current_pos)
                if start == -1:
                    break
                
                # 从左花括号开始，逐字符解析
                stack = []
                end = start
                
                for i in range(start, len(response)):
                    if response[i] == '{':
                        stack.append('{')
                    elif response[i] == '}':
                        stack.pop()
                        if not stack:  # 找到匹配的右花括号
                            end = i + 1
                            break
                
                if end > start:
                    try:
                        json_str = response[start:end]
                        score_data = json.loads(json_str)
                        
                        # 验证评分数据的格式：允许多个hamd评分
                        if all(isinstance(k, str) and isinstance(v, (int, float)) and k.startswith('hamd') for k, v in score_data.items()):
                            json_objects.append(score_data)
                        else:
                            print(f"无效的评分格式: {json_str}")
                            
                    except json.JSONDecodeError:
                        print(f"JSON解析错误: {json_str}")
                    except Exception as e:
                        print(f"处理评分数据时出错: {str(e)}")
                
                current_pos = end + 1
            
            # 如果找到有效的JSON对象
            if json_objects:
                # 如果只有一个评分对象，直接返回
                if len(json_objects) == 1:
                    return json_objects[0]
                # 如果有多个评分对象，合并它们
                else:
                    merged_scores = {}
                    for score_dict in json_objects:
                        merged_scores.update(score_dict)
                    return merged_scores
            
            return None
            
        except Exception as e:
            print(f"评分解析出错: {str(e)}")
            return None
    async def generate_chat_response(self, system_prompt, messages):
        """
        生成聊天回复
        
        Args:
            system_prompt: str, 系统提示词
            messages: list, 对话历史
            
        Returns:
            str: 生成的回复
        """
        try:
            # 构建完整的消息列表
            full_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages
            
            response = dashscope.Generation.call(
                model=self.model,
                messages=full_messages,
                result_format='message',
                **self.parameters
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise Exception(f"API调用失败: {response.code} - {response.message}")
                
        except Exception as e:
            print(f"生成回复时出错: {str(e)}")
            raise
