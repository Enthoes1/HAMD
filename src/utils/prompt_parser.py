class PromptParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.prompts = {}

    def parse_file(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                sections = content.split('#label#')
                
                for section in sections:
                    if section.strip():
                        # 获取label和提示词内容
                        lines = section.strip().split('\n', 1)
                        if len(lines) == 2:
                            label = lines[0].strip()
                            prompt_content = lines[1].strip()
                            self.prompts[label] = prompt_content
                            
        except FileNotFoundError:
            raise Exception(f"提示词文件未找到: {self.file_path}")
    
    def get_prompt(self, label):
        """获取特定标签的提示词"""
        return self.prompts.get(label)
    
    @staticmethod
    def get_question(prompt):
        """从提示词中提取问诊问题"""
        try:
            import re
            import json
            
            # 尝试直接解析JSON
            try:
                data = json.loads(prompt)
                if "条目详情" in data and "问题" in data["条目详情"]:
                    return data["条目详情"]["问题"]
            except json.JSONDecodeError:
                pass
            
            # 如果JSON解析失败，尝试使用正则表达式
            match = re.search(r'"问题"[^"]*"([^"]+)"', prompt)
            if match:
                return match.group(1)
                
            # 如果正则表达式也失败，尝试提取双引号之间的内容
            matches = re.findall(r'"([^"]+)"', prompt)
            for m in matches:
                if "您" in m or "你" in m or "最近" in m or "是否" in m:
                    return m
            
            return "请描述您的情况。"
        except Exception as e:
            print(f"提取问题出错: {str(e)}")
            return "请描述您的情况。"