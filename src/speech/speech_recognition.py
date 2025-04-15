from transformers import pipeline
import sounddevice as sd
import numpy as np
import torch
import os
import tempfile
import soundfile as sf
import base64
import io
import wave
import threading
from optimum.bettertransformer import BetterTransformer
from transformers import WhisperProcessor
from faster_whisper import WhisperModel

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
        print("CUDA不可用")
    print("=================\n")

class SpeechRecognition:
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SpeechRecognition, cls).__new__(cls)
            return cls._instance

    def __init__(self, model_name="Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper", use_auth_token=None):
        # 使用单例模式，确保模型只被加载一次
        with self.__class__._lock:
            if self.__class__._initialized:
                return
                
            print("初始化语音识别模型...")
            # 初始化时检查一次 GPU 状态
            check_gpu_status()
            
            # 判断使用CPU还是GPU
            compute_type = "float16" if torch.cuda.is_available() else "float32"
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            print(f"Using device: {device} for FasterWhisper model")
            
            try:
                # 使用FasterWhisper初始化模型
                self.model = WhisperModel(
                    model_name,
                    device=device,
                    compute_type=compute_type
                )
                
                print(f"Successfully loaded {model_name} model")
                
            except Exception as e:
                print(f"加载模型出错: {str(e)}")
                import traceback
                print(traceback.format_exc())
                raise
            
            self.recording = False
            self.sample_rate = 16000
            self.audio_data = []
            self.stream = None
            
            # 标记为已初始化
            self.__class__._initialized = True
        
    def start_recording(self):
        """开始录音"""
        try:
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
            
            if not self.audio_data:
                print("没有收到录音数据")
                return ""
            
            try:
                # 合并音频数据
                audio_data = np.concatenate(self.audio_data, axis=0)
                print(f"原始录音数据形状: {audio_data.shape}")
                
                # 确保音频是单通道的
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()  # 或者 audio_data = audio_data[:, 0]
                
                print(f"处理后的录音数据形状: {audio_data.shape}")
                
                # 清理GPU缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 使用 FasterWhisper 进行识别，禁用 VAD
                segments, info = self.model.transcribe(
                    audio_data, 
                    language="zh",
                    task="transcribe", 
                    beam_size=5,
                    vad_filter=False  # 禁用 VAD，避免 onnxruntime 依赖问题
                )
                
                # 提取文本
                text = " ".join([segment.text for segment in segments])
                return text.strip()
                
            except Exception as e:
                print(f"语音识别错误: {str(e)}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                return ""
            
        except Exception as e:
            print(f"录音处理错误: {str(e)}")
            return ""
    
    def process_audio(self, audio_data):
        """处理音频数据并返回识别结果"""
        try:
            # 解码 base64 数据
            wav_data = base64.b64decode(audio_data)
            
            # 使用 wave 模块读取音频数据
            with io.BytesIO(wav_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    # 获取音频参数
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    sample_rate = wav_file.getframerate()
                    
                    # 读取音频数据
                    audio_data = wav_file.readframes(wav_file.getnframes())
                    
                    # 转换为 numpy 数组
                    audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            # 确保音频是单通道的
            if len(audio_np.shape) > 1:
                audio_np = audio_np.mean(axis=1)
            
            print(f"音频数据形状: {audio_np.shape}")
            print(f"音频采样率: {sample_rate}")
            
            # 使用 FasterWhisper 进行识别，禁用 VAD
            segments, info = self.model.transcribe(
                audio_np, 
                language="zh",
                task="transcribe", 
                beam_size=5,
                vad_filter=False  # 禁用 VAD，避免 onnxruntime 依赖问题
            )
            
            # 提取文本
            text = " ".join([segment.text for segment in segments])
            return text.strip()
            
        except Exception as e:
            print(f"语音识别错误: {str(e)}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return ""
    
    def transcribe_audio(self, audio_data):
        """直接处理音频数据"""
        try:
            # 确保音频是单通道的
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()
            
            # 使用 FasterWhisper 进行识别，禁用 VAD
            segments, info = self.model.transcribe(
                audio_data, 
                language="zh",
                task="transcribe", 
                beam_size=5,
                vad_filter=False  # 禁用 VAD，避免 onnxruntime 依赖问题
            )
            
            # 提取文本
            text = " ".join([segment.text for segment in segments])
            return text.strip()
            
        except Exception as e:
            print(f"音频转写错误: {str(e)}")
            return ""
    
    def __del__(self):
        """析构函数，确保资源被正确释放"""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"关闭录音流错误: {str(e)}")