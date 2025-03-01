from openai import OpenAI
import re

class PatientAgent:
    def __init__(self, patient_id, model_config, mode=1):
        """初始化病人Agent
        
        Args:
            patient_id: 病人ID，必须符合'AI001'-'AI099'格式
            model_config: 模型配置
            mode: 对话历史模式，1=使用全部历史，2=只使用当前条目历史
        """
        if not re.match(r'AI0[0-9][0-9]', patient_id):
            raise ValueError("病人ID必须符合'AI001'-'AI099'格式")
            
        self.patient_id = patient_id
        self.conversation_history = []  # 全部对话历史
        self.current_item_history = []  # 当前条目的对话历史
        self.mode = mode
        self.total_tokens = 0  # token统计
        self.total_prompt_tokens = 0  # 提示词token统计
        self.total_completion_tokens = 0  # 回复token统计
        
        # 初始化OpenAI客户端，使用Ollama地址
        self.client = OpenAI(
            base_url=model_config['base_url'],  # Ollama地址
            api_key="not-needed"  # Ollama不需要API key
        )
        self.model = model_config['model']
        self.parameters = model_config.get('parameters', {})
        
    def clear_current_item_history(self):
        """清空当前条目的对话历史"""
        self.current_item_history = []
        
    async def generate_response(self, doctor_message):
        """根据医生的话生成回应"""
        # 记录医生的话
        self.conversation_history.append({
            'role': 'doctor',
            'content': doctor_message
        })
        self.current_item_history.append({
            'role': 'doctor',
            'content': doctor_message
        })
        
        try:
            # 构建消息列表
            messages = [
                {"role": "system", "content": "你要扮演一位轻症焦虑症患者，会有精神科大夫对你进行问诊。你的回答应该尽可能简短和口语化，最好每次不超过30个字。你的表述可以有不专业的地方，比如可以混淆抑郁和焦虑的区别等等"}
            ]
            
            # 根据模式选择使用的对话历史
            history = self.conversation_history if self.mode == 1 else self.current_item_history
            
            # 添加对话历史
            for msg in history:
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
            
            # 更新token统计
            usage = completion.usage
            if usage:
                self.total_prompt_tokens += usage.prompt_tokens
                self.total_completion_tokens += usage.completion_tokens
                self.total_tokens += usage.total_tokens
            
            response = completion.choices[0].message.content
            
            # 记录病人的回答
            self.conversation_history.append({
                'role': 'patient',
                'content': response
            })
            self.current_item_history.append({
                'role': 'patient',
                'content': response
            })
            
            return response
            
        except Exception as e:
            print(f"生成回答时出错: {str(e)}")
            return "对不起，我现在不太想说话..."  # 简化的降级回答
            
    def get_token_stats(self):
        """获取token使用统计"""
        return {
            'total_tokens': self.total_tokens,
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'mode': self.mode
        }