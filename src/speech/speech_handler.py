import edge_tts
import asyncio
import tempfile
import sounddevice as sd
import soundfile as sf

class SpeechHandler:
    def __init__(self):
        # 统一使用同一个语音设置
        self.voice = "zh-CN-XiaoxiaoNeural"
        
    async def text_to_speech(self, text, is_question=False):
        """
        将文字转换为语音并播放
        
        Args:
            text (str): 要转换的文本
            is_question (bool): 保留参数以维持接口兼容性
        """
        try:
            # 创建TTS通信对象
            communicate = edge_tts.Communicate(text, self.voice)
            
            # 保存并播放音频
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                await communicate.save(temp_file.name)
                audio_data, sample_rate = sf.read(temp_file.name)
                sd.play(audio_data, sample_rate)
                sd.wait()  # 等待播放完成
                
            return True
        except Exception as e:
            print(f"TTS错误: {str(e)}")
            return False