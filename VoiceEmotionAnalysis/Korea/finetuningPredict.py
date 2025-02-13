import os
import numpy as np
import torch
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
from transformers import AutoFeatureExtractor
from VoiceEmotionAnalysis.Korea.finetuned import MyLitModel  # MyLitModel은 실제 모델 파일 이름에 맞게 변경

# 기본 설정
SAMPLING_RATE = 16000
MODEL_DIR = 'finemodel'
audio_model_name = 'Rajaram1996/Hubert_emotion'
NUM_LABELS = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 감정 레이블 정의
emotion_labels = {
    0: "happiness",
    1: "angry",
    2: "disgust",
    3: "fear",
    4: "neutral",
    5: "sadness",
    6: "surprise"
}

# 특성 추출기 로드
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
audio_feature_extractor.return_attention_mask = True

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
    audio_values, audio_attn_mask = audio_feature['input_values'].to(DEVICE), audio_feature['attention_mask'].to(
        DEVICE)  # 데이터를 GPU로 이동
    with torch.no_grad():
        logits = model(audio_values, audio_attn_mask)
        preds = torch.argmax(logits, dim=1)
    return preds

# 체크포인트 파일 경로
checkpoint_path = os.path.join(MODEL_DIR, 'fold_0__epoch=03-val_acc=0.6621-train_acc=0.5000.ckpt')  # 실제 체크포인트 파일 이름

# 모델 불러오기
model = load_model(checkpoint_path)

# 녹음된 전체 데이터를 저장할 리스트
recorded_audio = []

# 실시간 녹음 및 예측
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    recorded_audio.extend(indata[:, 0])

# 입력 장치 확인
devices = sd.query_devices()
print(devices)

# 녹음 시작 및 종료 메시지 추가
print("녹음 시작!")
print("Press 'q' to stop recording.")
with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLING_RATE):
    while True:
        if input() == 'q':
            break
print("녹음 종료!")

# 전체 녹음된 파일 저장
audio_file_path = 'recorded_audio.wav'
sf.write(audio_file_path, np.array(recorded_audio), SAMPLING_RATE)
print(f"녹음된 파일이 {audio_file_path}로 저장되었습니다.")

# 5초마다 감정 예측
def predict_emotions_over_time(audio, segment_duration=5):
    segment_length = segment_duration * SAMPLING_RATE
    num_segments = len(audio) // segment_length
    emotions = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = audio[start:end]
        prediction = predict(model, segment)
        emotion = emotion_labels[prediction.item()]
        emotions.append(emotion)
        print(f'Segment {i + 1}: Predicted emotion: {emotion}')
    return emotions

# 감정 예측 수행
audio = np.array(recorded_audio)
emotions = predict_emotions_over_time(audio)

# 특징 시각화 함수
def plot_features(audio):
    times = np.arange(len(audio)) / SAMPLING_RATE  # 시간을 초 단위로 계산

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title('Audio Waveform')
    plt.plot(times, audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.specgram(audio, NFFT=2048, Fs=SAMPLING_RATE, noverlap=128)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()

# 전체 녹음된 파일의 특징 시각화
plot_features(audio)
