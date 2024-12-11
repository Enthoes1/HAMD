from openai import OpenAI
import re

class PatientAgent:
    def __init__(self, patient_id, model_config):
        """初始化病人Agent"""
        if not re.match(r'AI0[0-9][0-9]', patient_id):
            raise ValueError("病人ID必须符合'AI001'-'AI099'格式")
            
        self.patient_id = patient_id
        self.conversation_history = []
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=model_config['api_key'],
            base_url=model_config['base_url']
        )
        self.model = model_config['model']
        self.parameters = model_config.get('parameters', {})
        
    async def generate_response(self, doctor_message):
        """根据医生的话生成回应"""
        # 记录医生的话
        self.conversation_history.append({
            'role': 'doctor',
            'content': doctor_message
        })
        
        try:
            # 构建消息列表
            messages = [
                {"role": "system", "content": "你要扮演一位轻症抑郁症患者，会有精神科大夫对你进行问诊。你的回答应该尽可能简短和口语化，最好每次不超过30个字。你的表述可以有不专业的地方，比如可以混淆抑郁和焦虑的区别等等"}
            ]
            
            # 添加对话历史
            for msg in self.conversation_history:
                role = "assistant" if msg["role"] == "patient" else "user"
                messages.append({
                    "role": role,
                    "content": msg["content"]
                })
            
            # 调用API生成回答
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **self.parameters
            )
            
            response = completion.choices[0].message.content
            
            # 记录病人的回答
            self.conversation_history.append({
                'role': 'patient',
                'content': response
            })
            
            return response
            
        except Exception as e:
            print(f"生成回答时出错: {str(e)}")
            return "对不起，我现在不太想说话..."  # 简化的降级回答