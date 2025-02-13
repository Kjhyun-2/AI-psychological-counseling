import openai
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
api_key = os.getenv('OPENAI_API_KEY')
if api_key is None:
    raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")

# ChatOpenAI 인스턴스 생성
chat = ChatOpenAI(model_name='gpt-3.5-turbo', openai_api_key=api_key, max_tokens=4096)   #낮은 비용 max_tokens = 256

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

# 사용자 입력 예시
user_input = input("상담하고 싶은 내용을 적어주세요!: ")
""" 
입력한 예시

상담하고 싶은 내용을 적어주세요!: 안녕하세요 강지현입니다. 저는 올해 24살이고 지금은 연구실에서 캡스톤 작품을 만들기 위해  
이 시간동안 코드를 짜고 있어요 너무 지치고 힘듭니다. 어깨도 아프고 배도 고파요. 요즘은 너무 힘들어서 아무것도 하고싶지않네 
요. 어떻게 하면 좋을까요? 너무 우울하고 힘듭니다.
"""
emotion, response = process_user_input(user_input)
print("감정 상태:", emotion)
print("응답:", response)




import pyttsx3
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

# 음성 출력
text = emotion
engine.say(text)

# 대기하여 음성 재생이 끝날 때까지 기다림
engine.runAndWait()
