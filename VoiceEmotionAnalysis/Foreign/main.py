import os
import numpy as np
import torch
import librosa
import soundfile as sf
import sounddevice as sd
import wavio
from transformers import AutoFeatureExtractor
from VoiceEmotionRecognition import MyLitModel  # MyLitModel은 실제 모델 파일 이름에 맞게 변경

# 기본 설정
SAMPLING_RATE = 16000
MODEL_DIR = 'model'
audio_model_name = 'Rajaram1996/Hubert_emotion'
NUM_LABELS = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 감정 레이블 정의
emotion_labels = {
    0: "angry",
    1: "fear",
    2: "sad",
    3: "disgust",
    4: "neutral",
    5: "happy"
}

# 오디오 파일 불러오기
def load_audio(file_path, sampling_rate=SAMPLING_RATE):
    audio, _ = sf.read(file_path, dtype='float32')
    if len(audio.shape) > 1:  # 스테레오 파일을 모노로 변환
        audio = np.mean(audio, axis=1)
    if sampling_rate != SAMPLING_RATE:
        audio = librosa.resample(audio, sampling_rate, SAMPLING_RATE)
    return audio

# 음성을 녹음하고 WAV 파일로 저장하는 함수
def record_audio(filename, duration, samplerate=SAMPLING_RATE, device_index=None):
    print("녹음 시작...")
    if device_index is not None:
        sd.default.device = device_index
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # 녹음이 끝날 때까지 기다림
    print("녹음 완료!")
    wavio.write(filename, recording, samplerate, sampwidth=2)
    print(f"{filename} 파일로 저장되었습니다.")

# 특성 추출기 로드
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
audio_feature_extractor.return_attention_mask=True

# 모델 불러오기
def load_model(checkpoint_path):
    model = MyLitModel.load_from_checkpoint(
        checkpoint_path,
        audio_model_name=audio_model_name,
        num_labels=NUM_LABELS,
    )
    model.to(DEVICE)  # 모델을 GPU로 이동
    model.eval()
    return model

# 예측 함수
def predict(model, audio):
    audio_feature = audio_feature_extractor(raw_speech=audio, return_tensors='pt', sampling_rate=SAMPLING_RATE)
    audio_values, audio_attn_mask = audio_feature['input_values'].to(DEVICE), audio_feature['attention_mask'].to(DEVICE)  # 데이터를 GPU로 이동
    with torch.no_grad():
        logits = model(audio_values, audio_attn_mask)
        preds = torch.argmax(logits, dim=1)
    return preds

# 체크포인트 파일 경로
checkpoint_path = os.path.join(MODEL_DIR, 'fold_idx=0_epoch=03-val_acc=0.9084-train_acc=1.0000.ckpt')  # 실제 체크포인트 파일 이름

# 모델 불러오기
model = load_model(checkpoint_path)

# 입력 장치 확인
devices = sd.query_devices()
print(devices)

# 음성 녹음 및 저장
audio_file_path = '../Korea/test_recording.wav'  # 저장할 파일 경로
record_duration = 5  # 녹음 시간 (초)
input_device = 0  # 실제 시스템에서 사용할 입력 장치 인덱스로 변경

record_audio(audio_file_path, record_duration, device_index=input_device)

# 오디오 파일 로드
audio = load_audio(audio_file_path)

# 예측
prediction = predict(model, audio)
emotion = emotion_labels[prediction.item()]
print(f'Predicted emotion: {emotion}')
