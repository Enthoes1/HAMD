import whisper
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
from pathlib import Path
import torch
import os

def check_gpu_status():
    """检查GPU状态"""
    print("\n=== GPU 状态检查 ===")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
    print("=================\n")

class SpeechRecognition:
    def __init__(self, model_name="base"):  
        # 检查 ffmpeg 是否可用
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            print("ffmpeg 已安装并可用")
        except Exception as e:
            print("错误：ffmpeg 未安装或不可用")
            print("请使用 conda install ffmpeg 安装 ffmpeg")
            print(f"详细错误: {str(e)}")
            raise Exception("ffmpeg 未安装或不可用")
        
        # 检查GPU状态
        check_gpu_status()
        
        # 检查是否有可用的GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device} for Whisper model")
        
        try:
            # 使用项目目录作为模型缓存位置
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cache_dir = os.path.join(project_root, "models", "whisper")
            os.makedirs(cache_dir, exist_ok=True)
            print(f"使用项目目录作为模型缓存: {cache_dir}")
            
            self.model = whisper.load_model(
                model_name,
                device=self.device,
                download_root=cache_dir,  # 使用项目目录
                in_memory=True
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
            print("没有正在进行的录音")
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
                print(f"录音数据类型: {audio_data.dtype}")
                print(f"录音数据范围: {np.min(audio_data)} to {np.max(audio_data)}")
            except Exception as e:
                print(f"处理录音数据出错: {str(e)}")
                return ""
            
            try:
                # 直接使用内存中的音频数据，跳过文件保存
                try:
                    print(f"开始识别音频...")
                    # 确保音频数据是一维的
                    audio_data = audio_data.flatten()
                    result = self.model.transcribe(
                        audio_data,
                        language='zh'
                    )
                    print("识别完成")
                    
                    text = result["text"].strip()
                    print(f"识别结果: {text}")
                    return text
                    
                except Exception as e:
                    print(f"Whisper识别错误: {str(e)}")
                    import traceback
                    print(f"详细错误信息: {traceback.format_exc()}")
                    return ""
                    
            except Exception as e:
                print(f"录音处理错误: {str(e)}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                return ""
            
        except Exception as e:
            print(f"录音处理错误: {str(e)}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return ""
    
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"关闭录音流错误: {str(e)}")