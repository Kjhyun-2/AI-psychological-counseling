import pandas as pd
import os

# 한국어 감정 데이터 파일 경로
csv_paths = ['C:\\Users\\MMClab\\Desktop\\한국어 음성 감정\\감정 분류를 위한 대화 음성 데이터셋\\csvdata\\4차년도.csv',
             'C:\\Users\\MMClab\\Desktop\\한국어 음성 감정\\감정 분류를 위한 대화 음성 데이터셋\\csvdata\\5차년도.csv',
             'C:\\Users\\MMClab\\Desktop\\한국어 음성 감정\\감정 분류를 위한 대화 음성 데이터셋\\csvdata\\5차년도_2차.csv']

# 감정 라벨 매핑
emotion_to_label = {
    "happiness": 0,
    "angry": 1,
    "disgust": 2,
    "fear": 3,
    "neutral": 4,
    "sadness": 5,
    "surprise": 6
}

def determine_final_label(row):
    emotion_sums = {
        "happiness": 0,
        "angry": 0,
        "disgust": 0,
        "fear": 0,
        "neutral": 0,
        "sadness": 0,
        "surprise": 0
    }

    emotions = ['1번 감정', '2번 감정', '3번 감정', '4번 감정', '5번 감정']
    intensities = ['1번 감정세기', '2번 감정세기', '3번 감정세기', '4번 감정세기', '5번 감정세기']

    for emotion, intensity in zip(emotions, intensities):
        emotion_name = row[emotion].strip().lower()  # 감정 이름을 소문자로 변환
        intensity_value = row[intensity]
        #print(f"Emotion: {emotion_name}, Intensity: {intensity_value}")
        if emotion_name in emotion_sums:
            emotion_sums[emotion_name] += intensity_value

    #print(f"Emotion sums: {emotion_sums}")

    max_intensity = max(emotion_sums.values())
    if max_intensity == 0:
        return emotion_to_label["neutral"]

    final_emotion = max(emotion_sums, key=emotion_sums.get)
    return emotion_to_label[final_emotion]

def convert_to_training_format(csv_paths, output_path):
    all_data = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, encoding='cp949')
        df['label'] = df.apply(determine_final_label, axis=1)
        df['id'] = df['wav_id']
        df['path'] = df['wav_id'].apply(lambda x: f"./train/{x}.wav")

        all_data.append(df[['id', 'path', 'label']])

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(output_path, index=False)

# 변환된 데이터 저장 경로
output_path = 'C:\\Users\\MMClab\\Desktop\\한국어 음성 감정\\감정 분류를 위한 대화 음성 데이터셋\\csvdata\\train_data.csv'
convert_to_training_format(csv_paths, output_path)

print(f"Data conversion complete. The converted data is saved at {output_path}.")
