import os
from openai import OpenAI
import json

class LLMHandler:
    def __init__(self, model_config):
        self.client = OpenAI(
            api_key=model_config.get('api_key', os.getenv("DASHSCOPE_API_KEY")),
            base_url=model_config.get('base_url', "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self.model = model_config.get('model', 'qwen-plus')
        
    async def evaluate_response(self, prompt, user_response):
        try:
            # 构建完整提示词
            full_prompt = f"{prompt}\n\n用户回答：{user_response}"
            
            # 调用LLM进行评估
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {'role': 'user', 'content': full_prompt}
                ]
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