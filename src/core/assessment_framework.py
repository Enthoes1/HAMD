from utils.prompt_parser import PromptParser
from llm.llm_handler import LLMHandler

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
        current_item = self.items[self.current_item_index]
        
        # 调用LLM进行评估
        result = await self.llm_handler.evaluate_response(
            current_item.prompt, 
            user_response
        )
        
        # 记录对话历史
        self.conversation_history[current_item.item_id].append({
            'user': user_response,
            'assistant': result['data'] if result['type'] == 'message' else None
        })
        
        if result['type'] == 'score':
            # 如果是评分结果，存储并返回True表示该条目已完成
            self.scores[current_item.item_id] = result['data']
            return {'status': 'completed', 'score': result['data']}
        else:
            # 如果是普通对话，返回False表示继续对话
            return {'status': 'continue', 'message': result['data']}
        
    def next_item(self):
        if self.current_item_index < len(self.items) - 1:
            self.current_item_index += 1
            return self.items[self.current_item_index]
        return None
        
    def get_conversation_history(self, item_id):
        """获取特定条目的对话历史"""
        return self.conversation_history.get(item_id, [])