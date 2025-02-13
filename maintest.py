import sys
import os
import cv2
import numpy as np
from collections import defaultdict, deque
import torch
import torch.nn as nn
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils import face_utils
from torchvision.transforms import transforms
import dlib
from threading import Thread, Event
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import time

import sounddevice as sd
import soundfile as sf
import librosa
from transformers import AutoFeatureExtractor, HubertForSequenceClassification
import speech_recognition as sr  # SpeechRecognition 라이브러리 추가
import pyttsx3  # 텍스트를 음성으로 변환하기 위한 라이브러리

# 음성 합성 엔진 초기화
engine = pyttsx3.init()

# 모델 로딩을 위한 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))

# finetuned.py 파일이 있는 디렉토리를 PYTHONPATH에 추가
finetuned_dir = os.path.join(current_dir, 'VoiceEmotionAnalysis', 'Korea')
sys.path.append(finetuned_dir)

# 감정 인식 모델 경로
detection_model_path = os.path.join(current_dir, 'emotion_face', 'haarcascade_files', 'haarcascade_frontalface_default.xml')
emotion_model_path = os.path.join(current_dir, 'emotion_face', 'models', 'best_model-0.6522.hdf5')

# 눈 깜박임 모델 경로
blink_model_path = os.path.join(current_dir, 'eye_blink', 'best_model_2024_08_06_14_31_29.pth')
predictor_path = os.path.join(current_dir, 'eye_blink', 'shape_predictor_68_face_landmarks.dat')

# 음성 감정 모델 경로
MODEL_DIR = os.path.join(current_dir, 'VoiceEmotionAnalysis', 'Korea', 'finemodel')
checkpoint_path = os.path.join(MODEL_DIR, 'fold_0__epoch=25-val_acc=0.7307-train_acc=0.5000.ckpt')
audio_model_name = 'Rajaram1996/Hubert_emotion'
NUM_LABELS = 7

# eye_blink 폴더에서 모델을 가져오도록 경로 추가
sys.path.append(os.path.join(current_dir, 'eye_blink'))
from model import CNNModel  # 모델 정의를 import

# 감정 인식 모델 로딩
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# 눈 깜박임 모델 로딩
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
blink_model = CNNModel().to(device)
blink_model.load_state_dict(torch.load(blink_model_path, map_location=device))
blink_model.eval()
IMG_SIZE = (34, 26)

# 음성 감정 모델 로딩
audio_feature_extractor = AutoFeatureExtractor.from_pretrained(audio_model_name)
audio_feature_extractor.return_attention_mask = True

# 사전 학습된 모델 로딩
audio_model = HubertForSequenceClassification.from_pretrained(audio_model_name, num_labels=NUM_LABELS)
# 추가 학습된 가중치 로드
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = {k.replace("audio_model.", ""): v for k, v in checkpoint['state_dict'].items()}
audio_model.load_state_dict(state_dict)
audio_model.to(device)
audio_model.eval()

# 얼굴 검출기 및 랜드마크 예측기 초기화
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# transform 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 감정 집계용 딕셔너리
emotion_counts = defaultdict(int)

# 감정 인식 함수
def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))#64
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        emotion_counts[label] += 1

        return label
    return None

# 눈 깜박임 검출 파라미터
EYE_AR_THRESH = 0.22
EYE_AR_CONSEC_FRAMES = 1
COUNTER = 0
blink_count = 0
blink_times = []

# 눈 움직임 기록
left_eye_movements = []
right_eye_movements = []

# 실시간 녹음 및 예측
recording_event = Event()
recorded_audio = []

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 눈 깜박임 검출 함수
def detect_blink(frame, gray, shapes):
    global COUNTER, blink_count, blink_times, start_time

    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    leftEye = shapes[lStart:lEnd]
    rightEye = shapes[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0

    if ear < EYE_AR_THRESH:
        COUNTER += 1
    else:
        if COUNTER >= EYE_AR_CONSEC_FRAMES:
            blink_count += 1
            blink_times.append(time.time() - start_time)
        COUNTER = 0

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return frame

# 눈 움직임 추적 함수
def track_eye_movement(frame, gray, shapes):
    global left_eye_movements, right_eye_movements
    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)
    leftEye = shapes[lStart:lEnd]
    rightEye = shapes[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    left_eye_movements.append(leftEAR)
    right_eye_movements.append(rightEAR)

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

    return frame

def predict_audio_emotion(audio):
    audio_feature = audio_feature_extractor(raw_speech=audio, return_tensors='pt', sampling_rate=16000)
    audio_values, audio_attn_mask = audio_feature['input_values'].to(device), audio_feature['attention_mask'].to(device)
    with torch.no_grad():
        output = audio_model(audio_values, attention_mask=audio_attn_mask)
        logits = output.logits  # 로짓 값 추출
        preds = torch.argmax(logits, dim=1)
    return preds


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

# 실시간 녹음 및 예측
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    if recording_event.is_set():
        return
    recorded_audio.extend(indata[:, 0])

# 녹음 시작
def start_audio_recording():
    print("녹음 시작!")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        recording_event.wait()  # 녹음 종료 신호 대기
    print("녹음 종료!")

    # 녹음된 파일 저장
    audio_file_path = 'recorded_audio.wav'
    sf.write(audio_file_path, np.array(recorded_audio), 16000)
    print(f"녹음된 파일이 {audio_file_path}로 저장되었습니다.")

    # 녹음된 파일을 텍스트로 변환
    transcribed_text = transcribe_audio_to_text(audio_file_path)
    print(f"음성 인식 결과: {transcribed_text}")

    # 텍스트 파일로 저장
    with open('transcribed_text.txt', 'w', encoding='utf-8') as text_file:
        text_file.write(transcribed_text)
    print("텍스트 파일로 저장되었습니다.")

# 음성 파일을 텍스트로 변환하는 함수
def transcribe_audio_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="ko-KR")  # 한국어 인식
            return text
        except sr.UnknownValueError:
            return "음성을 인식할 수 없습니다."
        except sr.RequestError as e:
            return f"음성 인식 서비스에 문제가 발생했습니다: {e}"

# 5초마다 감정 예측
def predict_emotions_over_time(audio, segment_duration=5):
    segment_length = segment_duration * 16000
    num_segments = len(audio) // segment_length
    emotions = []
    for i in range(num_segments):
        start = i * segment_length
        end = start + segment_length
        segment = audio[start:end]
        prediction = predict_audio_emotion(segment)
        emotion = emotion_labels[prediction.item()]
        emotions.append(emotion)
        print(f'Segment {i + 1}: Predicted emotion: {emotion}')
    return emotions

# 오디오 특징 시각화
def plot_features(audio):
    times = np.arange(len(audio)) / 16000  # 시간을 초 단위로 계산

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title('Audio Waveform')
    plt.plot(times, audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.specgram(audio, NFFT=2048, Fs=16000, noverlap=128)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()

def main():
    cap = cv2.VideoCapture(0)  # 기본 카메라 사용
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    global start_time
    start_time = time.time()

    # 첫 번째 메시지 출력
    counter = 1
    engine.say(f"{counter}번")
    engine.runAndWait()

    # 음성 감정 예측 시작
    audio_thread = Thread(target=start_audio_recording)
    audio_thread.start()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray)

            for rect in rects:
                shapes = predictor(gray, rect)
                shapes = face_utils.shape_to_np(shapes)

                # 감정 인식
                emotion = detect_emotion(frame)
                if emotion:
                    cv2.putText(frame, f"Emotion: {emotion}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # 눈 깜박임 검출
                frame = detect_blink(frame, gray, shapes)

                # 눈 움직임 추적
                frame = track_eye_movement(frame, gray, shapes)

            cv2.imshow('Multi-function Stream', frame)

            # w 키를 누를 때마다 카운터 증가 및 음성 출력
            if cv2.waitKey(1) & 0xFF == ord('w'):
                counter += 1
                engine.say(f"{counter}번")
                engine.runAndWait()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                recording_event.set()  # 녹음 종료 신호 설정
                break

    except KeyboardInterrupt:
        print("Video stream interrupted.")

    cap.release()
    cv2.destroyAllWindows()

    # 녹음 종료 후 결과 분석
    audio_thread.join()
    audio = np.array(recorded_audio)
    emotions = predict_emotions_over_time(audio)

    # 음성 특징 시각화
    plot_features(audio)

    # 결과 그래프 표시
    plot_results()

def plot_results():
    global blink_times, left_eye_movements, right_eye_movements, emotion_counts

    plt.figure(figsize=(14, 7))

    # 눈 깜박임 그래프
    plt.subplot(2, 2, 1)
    if blink_times:
        total_duration = blink_times[-1] if blink_times else 0
        blink_record = np.zeros(int(total_duration) + 1)
        for t in blink_times:
            blink_record[int(t)] += 1
        plt.plot(blink_record, drawstyle='steps-pre')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Blink Count')
        plt.title('Blink Record Over Time')
    else:
        plt.text(0.5, 0.5, 'No blinks detected.', ha='center', va='center', fontsize=12)
        plt.axis('off')

    # 감정 분포 그래프
    plt.subplot(2, 2, 2)
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    if len(emotions) > 0 and sum(counts) > 0:
        plt.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=140)
    plt.title('Emotion Distribution')

    # 눈 움직임 그래프
    plt.subplot(2, 2, 3)
    plt.plot(left_eye_movements, label='Left Eye Movement')
    plt.plot(right_eye_movements, label='Right Eye Movement')
    plt.xlabel('Frame')
    plt.ylabel('EAR')
    plt.ylim(0.1, 0.5)
    plt.title('Eye Movement Tracking')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
