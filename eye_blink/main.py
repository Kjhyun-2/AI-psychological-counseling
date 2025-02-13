import sys
import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import time

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def run():
    EYE_AR_THRESH = 0.22  # 민감도를 높이기 위해 값 조정
    EYE_AR_CONSEC_FRAMES = 1  # 민감도를 높이기 위해 연속 프레임 수 감소

    COUNTER = 0
    blink_count = 0
    blink_times = []

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    start_time = time.time()

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

            cv2.imshow('Blink Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("비디오 스트림이 중단되었습니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return blink_times

def plot_blink_record(blink_times, total_duration):
    if not blink_times:
        print("No blinks detected.")
        return

    plt.figure(figsize=(10, 5))
    blink_record = np.zeros(int(total_duration) + 1)
    for t in blink_times:
        blink_record[int(t)] += 1

    plt.plot(blink_record, drawstyle='steps-pre')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Blink Count')
    plt.title('Blink Record Over Time')
    plt.show()

if __name__ == "__main__":
    blink_times = run()
    if blink_times:
        total_duration = blink_times[-1] if blink_times else 0
        plot_blink_record(blink_times, total_duration)