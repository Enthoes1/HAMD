import edge_tts
import asyncio
import tempfile
import sounddevice as sd
import soundfile as sf

class SpeechHandler:
    def __init__(self):
        # TTS设置
        self.voice = "zh-CN-XiaoxiaoNeural"
        
    async def text_to_speech(self, text):
        """将文字转换为语音并播放"""
        try:
            communicate = edge_tts.Communicate(text, self.voice)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                await communicate.save(temp_file.name)
                audio_data, sample_rate = sf.read(temp_file.name)
                sd.play(audio_data, sample_rate)
                sd.wait()  # 等待播放完成
                
            return True
        except Exception as e:
            print(f"TTS错误: {str(e)}")
            return False