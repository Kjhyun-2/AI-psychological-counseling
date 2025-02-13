from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from keras.models import load_model
from collections import defaultdict
import os

# 모델 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
detection_model_path = os.path.join(current_dir, 'haarcascade_files', 'haarcascade_frontalface_default.xml')
emotion_model_path = os.path.join(current_dir, 'models', 'best_model-0.6522.hdf5')#_mini_XCEPTION.102-0.66.hdf5

# 모델 존재 여부 확인
if not os.path.exists(detection_model_path):
    raise FileNotFoundError(f"Haar Cascade 파일이 {detection_model_path}에 존재하지 않습니다.")
if not os.path.exists(emotion_model_path):
    raise FileNotFoundError(f"감정 모델 파일이 {emotion_model_path}에 존재하지 않습니다.")

# 모델 로딩
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

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

        return label
    return None

def start_video_stream():
    emotion_counts = defaultdict(int)

    cap = cv2.VideoCapture(0)  # 기본 카메라 사용

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return emotion_counts

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            emotion = detect_emotion(frame)

            if emotion:
                emotion_counts[emotion] += 1


            cv2.imshow('Video Stream', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Video stream interrupted.")

    cap.release()
    cv2.destroyAllWindows()

    return emotion_counts
