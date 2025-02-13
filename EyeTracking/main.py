import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
import os
import matplotlib.font_manager as fm
font_path = "C:/Windows/Fonts/malgun.ttf"  # 윈도우 시스템에 설치된 한글 폰트 경로
fontprop = fm.FontProperties(fname=font_path)

# 파일 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
predictor_path = os.path.join(current_dir, 'shape_predictor_68_face_landmarks.dat')

# 얼굴 검출기와 눈 검출기를 초기화합니다.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)  # Dlib 얼굴 랜드마크 모델 경로

def eye_aspect_ratio(eye):
    # 눈의 6개 랜드마크 좌표를 사용하여 눈의 가로 세로 비율을 계산합니다.
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 왼쪽 눈과 오른쪽 눈의 랜드마크 인덱스
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# 비디오 스트림을 캡처합니다.
cap = cv2.VideoCapture(0)

# 눈 움직임을 기록할 리스트를 초기화합니다.
left_eye_movements = []
right_eye_movements = []

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            left_eye_movements.append(leftEAR)
            right_eye_movements.append(rightEAR)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except KeyboardInterrupt:
    print("비디오 스트림이 중단되었습니다.")

cap.release()
cv2.destroyAllWindows()

# 눈 움직임을 선 그래프로 표현합니다.
plt.figure(figsize=(14, 7))
plt.plot(left_eye_movements, label='Left Eye Movement')
plt.plot(right_eye_movements, label='Right Eye Movement')
plt.xlabel('Frame', fontproperties=fontprop)
plt.ylabel('눈 움직임', fontproperties=fontprop)
plt.ylim(0.1, 0.5)  # y축 범위를 0.1에서 0.5로 고정
plt.title('눈 움직임 감지', fontproperties=fontprop)
plt.legend(prop=fontprop)
plt.show()
