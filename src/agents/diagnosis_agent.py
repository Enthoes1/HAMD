from src.core.assessment_framework import AssessmentFramework
from src.utils.prompt_parser import PromptParser

class DiagnosisAgent:
    def __init__(self, prompt_file_path, model_config, patient_agent=None):
        """初始化问诊Agent
        
        Args:
            prompt_file_path: 提示词文件路径
            model_config: 模型配置
            patient_agent: 病人Agent的引用，用于在切换条目时清空历史
        """
        self.framework = AssessmentFramework(prompt_file_path, model_config)
        self.current_question = None
        self.conversation_history = {}  # 每个条目的对话历史
        self.patient_agent = patient_agent
        
    def set_patient_info(self, patient_info):
        """设置病人信息，这是保存进度所必需的"""
        self.framework.patient_info = patient_info
        
    async def get_next_response(self, user_input=None):
        """获取下一个回应"""
        try:
            if user_input is None:
                # 获取新问题
                current_item = self.framework.items[self.framework.current_item_index]
                question = self.framework.prompt_parser.get_question(current_item.prompt)
                self.current_question = question
                
                # 初始化当前条目的对话历史
                if current_item.item_id not in self.conversation_history:
                    self.conversation_history[current_item.item_id] = []
                
                return question
            else:
                # 获取当前条目
                current_item = self.framework.items[self.framework.current_item_index]
                
                # 获取当前条目的对话历史
                history = self.conversation_history.get(current_item.item_id, [])
                
                # 处理用户输入，获取LLM响应
                result = await self.framework.process_response(
                    user_input,
                    history,
                    self.current_question
                )
                
                if isinstance(result, dict):
                    if result.get('type') == 'score':
                        # 存储评分
                        scores = result.get('data')
                        if not isinstance(scores, list):
                            scores = [scores]
                        for score in scores:
                            self.framework.scores[current_item.item_id] = score
                        
                        # 切换到下一个条目
                        next_item = self.framework.next_item()
                        if next_item:
                            # 如果有病人Agent，清空其当前条目历史
                            if self.patient_agent:
                                self.patient_agent.clear_current_item_history()
                                
                            # 自动获取下一个问题
                            question = self.framework.prompt_parser.get_question(next_item.prompt)
                            self.current_question = question
                            return question
                        else:
                            # 评估完成
                            self.framework.save_assessment_result()
                            return None
                            
                    elif result.get('type') == 'message':
                        # 更新对话历史
                        history.append({
                            'user': user_input,
                            'assistant': result.get('data', '')
                        })
                        self.conversation_history[current_item.item_id] = history
                        return result.get('data', '')
                        
                return result
                    
        except Exception as e:
            print(f"Agent响应错误: {str(e)}")
            return f"系统错误: {str(e)}" 