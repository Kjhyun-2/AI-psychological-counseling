import sys
import os
import matplotlib.pyplot as plt

# 현재 파일의 디렉토리 경로를 얻음
current_dir = os.path.dirname(os.path.abspath(__file__))

# 'emotion_face' 디렉토리의 경로를 얻음
emotion_face_dir = os.path.join(current_dir, 'emotion_face')

# 'emotion_face' 디렉토리를 PYTHONPATH에 추가
sys.path.append(emotion_face_dir)

from real_time_video import start_video_stream

def plot_emotion_counts(emotion_counts):
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())

    # Pie chart
    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    plt.pie(counts, labels=emotions, autopct='%1.1f%%', startangle=140)
    plt.title('Emotion Distribution')

    # Bar chart
    plt.subplot(1, 2, 2)
    plt.bar(emotions, counts, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.xlabel('Emotions')
    plt.ylabel('Counts')
    plt.title('Emotion Distribution')

    plt.tight_layout()
    plt.show()

def print_emotion_counts(emotion_counts):
    print("\nFinal Emotion Counts:")
    for emotion, count in emotion_counts.items():
        print(f"{emotion}: {count} times")


if __name__ == '__main__':
    try:
        # 실시간 감정 인식 실행 및 감정 카운트 반환
        emotion_counts = start_video_stream()
        # 최종감정 카운트
        print_emotion_counts(emotion_counts)

        # 감정 카운트 그래프 출력
        plot_emotion_counts(emotion_counts)
    except KeyboardInterrupt:
        print("프로그램이 종료되었습니다.")
        sys.exit()
