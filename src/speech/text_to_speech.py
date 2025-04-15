import asyncio
import edge_tts
import tempfile
import os
import base64
from typing import Optional

class TextToSpeech:
    def __init__(self):
        self.voice = "zh-CN-XiaoxiaoNeural"  # 默认使用中文女声
        
    def speak(self, text: str) -> Optional[str]:
        """将文本转换为语音并返回 base64 编码的音频数据"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
                
            print(f"生成语音临时文件: {temp_path}")
            
            # 使用 edge-tts 生成语音
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            communicate = edge_tts.Communicate(text, self.voice)
            loop.run_until_complete(communicate.save(temp_path))
            loop.close()
            
            print(f"语音生成完成，文件大小: {os.path.getsize(temp_path)} 字节")
            
            # 读取音频文件并转换为 base64
            with open(temp_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                base64_audio = base64.b64encode(audio_data).decode('utf-8')
            
            # 清理临时文件
            os.unlink(temp_path)
            
            return base64_audio
            
        except Exception as e:
            print(f"语音合成错误: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None 