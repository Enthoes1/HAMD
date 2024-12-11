from src.utils.prompt_parser import PromptParser
from src.llm.llm_handler import LLMHandler
import json
import os
from datetime import datetime
from src.speech.speech_handler import SpeechHandler

class AssessmentItem:
    def __init__(self, item_id, prompt):
        self.item_id = item_id
        self.prompt = prompt
        self.score = None

class AssessmentFramework:
    def __init__(self, prompt_file_path, model_config):
        self.items = []
        self.current_item_index = 0
        self.scores = {}  # 存储评分
        self.score_history = {}  # 评分历史
        self.conversation_history = {}  # 存储每个条目的对话历史
        self.patient_info = {}  # 存储患者基本信息
        
        # 创建评分结果保存目录
        self.results_dir = os.path.join(os.path.dirname(prompt_file_path), "assessment_results")
        self.progress_dir = os.path.join(os.path.dirname(prompt_file_path), "progress")
        
        # 创建必要的目录
        for directory in [self.results_dir, self.progress_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # 初始化提示词解析器和LLM处理器
        self.prompt_parser = PromptParser(prompt_file_path)
        self.prompt_parser.parse_file()
        self.llm_handler = LLMHandler(model_config)
        self.speech_handler = SpeechHandler()
        
    def initialize_items_from_prompts(self):
        """根据提示词文件初始化评估项目"""
        for label, prompt in self.prompt_parser.prompts.items():
            item = AssessmentItem(
                item_id=label,
                prompt=prompt
            )
            self.add_item(item)
            self.conversation_history[label] = []
        
    def add_item(self, item):
        self.items.append(item)
        
    async def process_response(self, user_response, history=None, question=None):
        try:
            current_item = self.items[self.current_item_index]
            
            result = await self.llm_handler.evaluate_response(
                current_item.prompt, 
                user_response,
                history,
                question
            )
            
            # 记录用户的输入和LLM的响应
            if not self.conversation_history[current_item.item_id]:
                self.conversation_history[current_item.item_id] = []
            
            # 存储对话历史，使用原始响应
            history_entry = {
                'user': user_response,
                'llm_response': result.get('raw_response', result['data']),  # 优先使用原始响应
                'show_response': result.get('show_response', True),
                'type': result['type']
            }
            self.conversation_history[current_item.item_id].append(history_entry)
            
            if result['type'] == 'score':
                # 获取评分数据
                scores = result['data']
                if not isinstance(scores, list):
                    scores = [scores]
                    
                # 更新评分
                for score_dict in scores:
                    for label, value in score_dict.items():
                        self.scores[label] = value
                        
                        # 添加到评分历史
                        if label not in self.score_history:
                            self.score_history[label] = []
                        self.score_history[label].append({
                            'score': value,
                            'timestamp': datetime.now().isoformat()
                        })
            
            return result
            
        except Exception as e:
            print(f"处理响应时出错: {str(e)}")
            raise
        
    def next_item(self):
        if self.current_item_index < len(self.items) - 1:
            self.current_item_index += 1
            return self.items[self.current_item_index]
        return None
        
    def get_conversation_history(self, item_id):
        """获取特定条目的对话历史"""
        return self.conversation_history.get(item_id, [])
        
    def set_patient_info(self, info):
        """设置患者基本信息"""
        self.patient_info = info
        
    def save_assessment_result(self):
        """保存评估结果到文件"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"assessment_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            # 在评估数据中加入患者信息
            assessment_data = {
                'timestamp': timestamp,
                'patient_info': self.patient_info,  # 添加患者信息
                'scores': self.scores,  # 最终评分
                'score_history': self.score_history,  # 评分历史
                'conversation_history': self.conversation_history
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(assessment_data, f, ensure_ascii=False, indent=2)
                
            print(f"评估结果已保存到: {filepath}")
            
        except Exception as e:
            print(f"保存评估结果出错: {str(e)}")
    
    def save_progress(self):
        """保存当前进度"""
        try:
            if not self.patient_info:
                print("没有患者信息，无法保存进度")
                return False
            
            if 'id' not in self.patient_info:
                print(f"患者信息中缺少ID字段: {self.patient_info}")
                return False
            
            # 打印当前状态
            print(f"正在保存进度:")
            print(f"- 患者ID: {self.patient_info['id']}")
            print(f"- 当前题目: {self.current_item_index + 1}")
            print(f"- 已完成评分: {len(self.scores)}")
            
            progress_data = {
                'patient_info': self.patient_info,
                'current_item_index': self.current_item_index,
                'scores': self.scores,
                'score_history': self.score_history,
                'conversation_history': self.conversation_history
            }
            
            filename = f"progress_{self.patient_info['id']}.json"
            filepath = os.path.join(self.progress_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2)
                
            print(f"进度已保存: {filepath}")
            return True
            
        except Exception as e:
            print(f"保存进度失败: {str(e)}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    def load_progress(self, patient_id):
        """加载已保存的进度"""
        try:
            filename = f"progress_{patient_id}.json"
            filepath = os.path.join(self.progress_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"未找到进度文件: {filepath}")
                return False
                
            with open(filepath, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
                
            # 恢复状态
            self.patient_info = progress_data['patient_info']
            self.current_item_index = progress_data['current_item_index']
            self.scores = progress_data['scores']
            self.score_history = progress_data['score_history']
            self.conversation_history = progress_data['conversation_history']
            
            print(f"已恢复进度，当前题目: {self.current_item_index + 1}")
            return True
            
        except Exception as e:
            print(f"加载进度失败: {str(e)}")
            return False