import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
from pathlib import Path
import torch

def check_gpu_status():
    """检查GPU状态"""
    print("\n=== GPU 状态检查 ===")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
    print("=================\n")

class SpeechRecognition:
    def __init__(self, model_name="large-v3"):  # 使用 medium 型号平衡性能和准确率
        # 检查GPU状态
        check_gpu_status()
        
        # 检查是否有可用的GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device} for Whisper model")
        
        try:
            # 使用 weights_only=True 加载模型
            self.model = whisper.load_model(
                model_name,
                device=self.device,
                download_root=None,  # 使用默认下载路径
                in_memory=True      # 保持模型在内存中
            ).to(self.device)
            print(f"Successfully loaded {model_name} model")
        except Exception as e:
            print(f"加载模型出错: {str(e)}")
            raise
        
        self.recording = False
        self.sample_rate = 16000
        self.audio_data = []
        self.stream = None
        
    def start_recording(self):
        """开始录音"""
        try:
            # 确保之前的录音已经停止
            if self.stream is not None:
                try:
                    self.stream.stop()
                    self.stream.close()
                except:
                    pass
                
            self.recording = True
            self.audio_data = []
            
            def callback(indata, frames, time, status):
                if status:
                    print(f"录音状态: {status}")
                if self.recording:
                    self.audio_data.append(indata.copy())
            
            # 开始录音
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=callback,
                dtype=np.float32
            )
            print("开始录音...")
            self.stream.start()
            
        except Exception as e:
            print(f"开始录音时出错: {str(e)}")
            self.recording = False
            raise
    
    def stop_recording(self):
        """停止录音并返回识别结果"""
        if not self.recording or not self.stream:
            return ""
            
        try:
            self.recording = False
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
            # 检查是否有录音数据
            if not self.audio_data:
                print("没有收到录音数据")
                return ""
                
            # 将录音数据转换为numpy数组
            try:
                audio_data = np.concatenate(self.audio_data, axis=0)
                print(f"录音数据形状: {audio_data.shape}")
            except Exception as e:
                print(f"处理录音数据出错: {str(e)}")
                return ""
                
            # 创建临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # 保存音频文件
                sf.write(temp_path, audio_data, self.sample_rate)
                
                # 使用whisper进行识别，使用最基本的配置
                result = self.model.transcribe(
                    temp_path,
                    language='zh'  # 只指定语言为中文
                )
                
                return result["text"].strip()
                
            finally:
                # 确保删除临时文件
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception as e:
                    print(f"删除临时文件失败: {str(e)}")
                    
        except Exception as e:
            print(f"录音处理错误: {str(e)}")
            return ""
        
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"关闭录音流错误: {str(e)}")