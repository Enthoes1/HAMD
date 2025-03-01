from src.utils.prompt_parser import PromptParser
from src.llm.llm_handler import LLMHandler
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
        """初始化评估框架"""
        self.items = []
        self.current_item_index = 0
        self.scores = {}  # 存储评分，使用hamd1-hamd24格式
        self.score_history = {}  # 评分历史
        self.conversation_history = {}  # 存储每个条目的对话历史
        self.patient_info = {}  # 存储患者基本信息
        self.insight_item = None  # 存储自知力评估项目
        
        # 获取项目根目录
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # 创建评分结果保存目录
        self.results_dir = os.path.join(self.root_dir, "assessment_results")
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        self.progress_dir = os.path.join(self.root_dir, "progress")
        if not os.path.exists(self.progress_dir):
            os.makedirs(self.progress_dir)
            
        # 初始化提示词解析器和LLM处理器
        self.prompt_parser = PromptParser(prompt_file_path)
        self.llm_handler = LLMHandler(model_config)
        
        # 按数字排序解析提示词
        self.prompt_parser.parse_file(sort_by_number=False)
        
    def initialize_items_from_prompts(self):
        """根据提示词文件初始化评估项目"""
        for label, prompt in self.prompt_parser.prompts.items():
            item = AssessmentItem(
                item_id=label,
                prompt=prompt
            )
            # 如果是自知力评估项目（hamd17），单独存储
            if "hamd17" in prompt:
                self.insight_item = item
                continue
            self.add_item(item)
            self.conversation_history[label] = []
        
        # 如果存在自知力评估项目，将其添加到列表末尾
        if self.insight_item:
            self.conversation_history[self.insight_item.item_id] = []
            self.add_item(self.insight_item)
            
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
            
            # 存储用户回应
            history_entry = {
                'content': user_response,
                'role': 'patient'
            }
            self.conversation_history[current_item.item_id].append(history_entry)
            
            # 存储LLM响应
            llm_entry = {
                'content': result.get('raw_response', result['data']),  # 优先使用原始响应
                'role': 'assistant',
                'show_response': result.get('show_response', True),
                'type': result['type']
            }
            self.conversation_history[current_item.item_id].append(llm_entry)
            
            if result['type'] == 'score':
                # 获取评分数据
                scores = result['data']
                if not isinstance(scores, list):
                    scores = [scores]
                    
                # 更新评分，直接使用hamd格式
                for score_dict in scores:
                    for hamd_label, value in score_dict.items():
                        # 直接使用hamd格式的标签
                        self.scores[hamd_label] = value
                        
                        # 添加到评分历史
                        if hamd_label not in self.score_history:
                            self.score_history[hamd_label] = []
                        self.score_history[hamd_label].append({
                            'score': value,
                            'timestamp': datetime.now().isoformat()
                        })
            
            return result
            
        except Exception as e:
            print(f"处理响应时出错: {str(e)}")
            raise
        
    def next_item(self):
        if self.current_item_index < len(self.items) - 1:
            next_item = self.items[self.current_item_index + 1]
            
            # 如果下一个是自知力评估项目，先检查总分
            if "hamd17" in next_item.prompt:
                # 计算除hamd17外的所有评分总和
                total_score = sum(score for label, score in self.scores.items() if label != "hamd17")
                
                # 如果总分小于等于8分，直接设置hamd17为0分并继续评估
                if total_score <= 8:
                    self.scores["hamd17"] = 0
                    if "hamd17" not in self.score_history:
                        self.score_history["hamd17"] = []
                    self.score_history["hamd17"].append({
                        'score': 0,
                        'timestamp': datetime.now().isoformat()
                    })
                    # 保存进度
                    self.save_progress()
                    # 继续到下一个项目
                    self.current_item_index += 1
                    if self.current_item_index < len(self.items) - 1:
                        return self.items[self.current_item_index + 1]
                    else:
                        # 如果是最后一个项目，保存结果并返回None
                        self.save_assessment_result()
                        return None
                    
            self.current_item_index += 1
            return next_item
        
        # 如果是最后一个项目，保存结果
        if self.current_item_index == len(self.items) - 1:
            self.save_assessment_result()
        return None
        
    def get_conversation_history(self, item_id):
        """获取特定条目的对话历史"""
        return self.conversation_history.get(item_id, [])
        
    def set_patient_info(self, info):
        """设置患者基本信息"""
        self.patient_info = info
        
    def save_assessment_result(self):
        """保存评估结果（仅在评估完全完成时调用）"""
        try:
            if not self.patient_info:
                print("警告：没有患者信息，无法保存评估结果")
                return False
                
            # 检查是否有评分数据
            if not self.scores:
                print("警告：没有评分数据，无法保存评估结果")
                return False
                
            # 使用患者ID和时间戳生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"assessment_{timestamp}.json"
            filepath = os.path.join(self.results_dir, filename)
            
            print(f"准备保存结果到: {filepath}")
            print(f"当前评分数据: {self.scores}")
            
            # 准备保存的数据
            result_data = {
                "timestamp": timestamp,
                "patient_info": self.patient_info,
                "scores": self.scores,
                "conversation_history": self.conversation_history
            }
            
            # 保存结果
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                
            print(f"评估结果已保存到: {filepath}")
            return True
            
        except Exception as e:
            print(f"保存评估结果时出错: {str(e)}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    def save_progress(self):
        """保存当前进度（在每次评分或对话更新后调用）"""
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
                'conversation_history': self.conversation_history,
                'last_update': datetime.now().isoformat()  # 添加最后更新时间
            }
            
            filename = f"progress_{self.patient_info['id']}.json"
            filepath = os.path.join(self.progress_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, ensure_ascii=False, indent=2, separators=(',', ': '))
                
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