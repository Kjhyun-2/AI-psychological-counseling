import openai
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import pyttsx3

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

# ChatOpenAI 인스턴스 생성
chat = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=api_key, max_tokens=4096)   # 낮은 비용 max_tokens = 256

def analyze_emotion(user_input):
    prompt_for_emotion = f"다음 내용을 읽고 사용자의 감정 상태를 매우 상세하고 30줄 정도로 자세하게 분석하세요:\n---\n{user_input}\n---\n"
    response = chat.invoke(prompt_for_emotion)
    emotion_state = response.content.strip()
    return emotion_state

def generate_empathetic_response(emotion_state):
    prompt_for_reply = f"사용자의 감정 상태는 '{emotion_state}'입니다. 이에 대한 공감과 지지를 매우 상세하고 30줄 정도로 길게 표현하는 답변을 작성하세요."
    response = chat.invoke(prompt_for_reply)
    empathetic_response = response.content.strip()
    return empathetic_response

def process_user_input(user_input):
    emotion_state = analyze_emotion(user_input)
    empathetic_response = generate_empathetic_response(emotion_state)
    return emotion_state, empathetic_response

# transcribed_text.txt 파일에서 텍스트 읽기
transcribed_text_path = 'C:\\Users\\wlgus\\PycharmProjects\\capstone\\capstone\\transcribed_text.txt'

if not os.path.exists(transcribed_text_path):
    raise FileNotFoundError(f"{transcribed_text_path} 파일을 찾을 수 없습니다.")

with open(transcribed_text_path, 'r', encoding='utf-8') as file:
    user_input = file.read().strip()

# 사용자 입력을 처리하여 감정 상태와 응답 생성
emotion, response = process_user_input(user_input)
print("감정 상태:", emotion)
print("응답:", response)

"""
# pyttsx3 엔진 초기화
engine = pyttsx3.init()
# 사용 가능한 음성 목록 가져오기
voices = engine.getProperty('voices')
# 여성 목소리로 설정
for voice in voices:
    if "female" in voice.name.lower():
        engine.setProperty('voice', voice.id)
        break
# 음성 속성 설정 (optional)
engine.setProperty('rate', 150)  # 말하는 속도
engine.setProperty('volume', 1)  # 볼륨 (0.0에서 1.0 사이)

# 감정 상태와 응답을 음성으로 출력
engine.say("감정 상태는 다음과 같습니다.")
engine.say(emotion)
engine.say("이에 대한 응답은 다음과 같습니다.")
engine.say(response)

# 대기하여 음성 재생이 끝날 때까지 기다림
engine.runAndWait()
"""