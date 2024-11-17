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