import pyaudio
import webrtcvad
import collections
import wave
import threading
import numpy as np
import soundfile as sf
import whisper
import torch
from concurrent.futures import ThreadPoolExecutor
import signal
import sys

# 根据可用性选择使用CUDA或CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# 加载 Whisper 模型
model = whisper.load_model("base").to(device).eval()

# 设置参数
FORMAT = pyaudio.paInt16  # 采样位数
CHANNELS = 1              # 单声道
RATE = 16000              # 采样率
CHUNK = 320               # 数据块大小
KEYWORDS = ["土豆", "你好土豆","hi tudou"]      # 关键词列表
RECORD_SECONDS = 5        # 录音时长（秒）
VAD_MODE = 3              # VAD模式（0-3，3最敏感）

def vad_collector(sample_rate, channels, vad, padding_duration=0.2, max_speech_duration=1.0):
    """
    使用VAD收集语音片段。

    参数:
    sample_rate (int): 采样率。
    channels (int): 声道数。
    vad (webrtcvad.Vad): VAD实例。
    padding_duration (float): 填充持续时间，默认为0.2秒。
    max_speech_duration (float): 最大语音持续时间，默认为1.0秒。

    生成:
    b''.join(voiced_frames) (bytes): 语音片段的字节数据。
    """
    num_padding_frames = int(padding_duration * sample_rate / CHUNK)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    try:
        while True:
            chunk = stream.read(CHUNK)
            is_speech = vad.is_speech(chunk, sample_rate)
            if not triggered:
                ring_buffer.append((chunk, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                voiced_frames.append(chunk)
                ring_buffer.append((chunk, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    yield b''.join(voiced_frames)
                    ring_buffer.clear()
                    voiced_frames = []
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def transcribe_and_spot_keywords(audio_data):
    """
    转录音频数据并识别关键词。

    参数:
    audio_data (bytes): 音频数据的字节流。
    """
    try:
        min_audio_length = RATE * 0.5  # 至少需要0.5秒的音频数据
        if len(audio_data) < min_audio_length:
            print("Audio data is too short, skipping transcription.")
            return

        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        result = model.transcribe(audio_np)
        transcription = result["text"]
        print(f"Transcription: {transcription}")
        detected_keywords = [kw for kw in KEYWORDS if kw.lower() in transcription.lower()]
        if detected_keywords:
            print(f"Detected Keywords: {detected_keywords}")
        else:
            print("No keywords detected.")
    except Exception as e:
        print(f"Error during transcription: {e}")

def listen_and_detect():
    """
    监听语音输入并检测关键词。
    """
    vad = webrtcvad.Vad(VAD_MODE)
    with ThreadPoolExecutor(max_workers=5) as executor:
        for audio_data in vad_collector(RATE, CHANNELS, vad):
            executor.submit(transcribe_and_spot_keywords, audio_data)

if __name__ == "__main__":
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    listen_and_detect()
