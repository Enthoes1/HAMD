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
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    else:
        print("CUDA不可用，可能的原因：")
        print("1. PyTorch未安装CUDA版本")
        print("2. NVIDIA驱动未正确安装")
        print("3. CUDA工具包未正确安装")
        print("\n建议执行以下步骤：")
        print("1. 确认是否安装了NVIDIA驱动")
        print("2. 使用 nvidia-smi 命令检查GPU状态")
        print("3. 重新安装支持CUDA的PyTorch版本：")
        print("   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
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
                
                # 确保音频数据是一维的
                audio_data = audio_data.flatten()
                
                # 如果使用GPU，将数据转移到GPU
                if torch.cuda.is_available():
                    with torch.cuda.device(0):  # 使用第一个GPU
                        torch.cuda.empty_cache()  # 清理GPU缓存
                
            except Exception as e:
                print(f"处理录音数据出错: {str(e)}")
                return ""
            
            try:
                # 使用whisper进行语音识别
                result = self.model.transcribe(
                    audio_data,
                    language='zh',          # 指定中文
                    task='transcribe',      # 使用转写任务
                    fp16=True if torch.cuda.is_available() else False,  # 使用半精度
                    beam_size=1,            # 减少内存使用
                    best_of=1,              # 减少内存使用
                    temperature=0.0,        # 减少随机性
                    compression_ratio_threshold=2.4,
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False  # 减少内存使用
                )
                
                text = result["text"].strip()
                return text
                
            except Exception as e:
                print(f"Whisper识别错误: {str(e)}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                return ""
            
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