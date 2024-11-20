from utils.prompt_parser import PromptParser
from llm.llm_handler import LLMHandler
import json
import os
from datetime import datetime

class AssessmentItem:
    def __init__(self, item_id, prompt):
        self.item_id = item_id
        self.prompt = prompt
        self.score = None

class AssessmentFramework:
    def __init__(self, prompt_file_path, model_config):
        self.items = []
        self.current_item_index = 0
        self.scores = {}
        self.conversation_history = {}  # 存储每个条目的对话历史
        
        # 创建评分结果保存目录
        self.results_dir = os.path.join(os.path.dirname(prompt_file_path), "assessment_results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # 初始化提示词解析器和LLM处理器
        self.prompt_parser = PromptParser(prompt_file_path)
        self.prompt_parser.parse_file()
        self.llm_handler = LLMHandler(model_config)
        
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
        
    async def process_response(self, user_response):
        try:
            current_item = self.items[self.current_item_index]
            history = self.conversation_history[current_item.item_id]
            
            result = await self.llm_handler.evaluate_response(
                current_item.prompt, 
                user_response,
                history
            )
            
            if result['type'] == 'score':
                # 存储评分结果
                self.scores[current_item.item_id] = result['data']
                # 每次有新的评分就保存
                self.save_assessment_result()
            else:
                # 记录对话历史
                self.conversation_history[current_item.item_id].append({
                    'user': user_response,
                    'assistant': result['data']
                })
            
            return result
        except Exception as e:
            print(f"处理响应出错: {str(e)}")
            raise
        
    def next_item(self):
        if self.current_item_index < len(self.items) - 1:
            self.current_item_index += 1
            return self.items[self.current_item_index]
        return None
        
    def get_conversation_history(self, item_id):
        """获取特定条目的对话历史"""
        return self.conversation_history.get(item_id, [])
        
    def save_assessment_result(self):
        """保存评估结果到文件"""
        try:
            # 生成时间戳作为文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"assessment_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            # 构建完整的评估数据
            assessment_data = {
                'timestamp': timestamp,
                'scores': self.scores,
                'conversation_history': self.conversation_history
            }
            
            # 保存到文件
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(assessment_data, f, ensure_ascii=False, indent=2)
                
            print(f"评估结果已保存到: {filepath}")
            
        except Exception as e:
            print(f"保存评估结果出错: {str(e)}")