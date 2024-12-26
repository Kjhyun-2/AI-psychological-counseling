<h1>
  <img src="https://github.com/user-attachments/assets/df00160b-8a4f-4bae-9bbd-ce39565723f8" alt="Logo" width="35">
  AI-psychological-counseling
</h1>

## 📑 Content

1. [📖 About](#-about)
2. [🔎 Features](#-features)
3. [🗣 Service](#-Service)
4. [Team](#team)
5. [Install](#install)
6. [🛠 Tool](#tool)
---
   
## 🎧 AI를 활용한 심리상담 웹사이트 

현대인들은 복잡한 사회 구조와 스트레스 속에서 심리적 문제를 겪는 경우가 많아지고 있으며, 이에 따라 심리 상담에 대한 수요도 증가하고 있습니다. 하지만 전통적인 상담 서비스는 비용과 접근성의 한계로 인해 많은 사람들이 적절한 도움을 받지 못하고 있습니다. 이러한 문제를 해결하기 위해, 익명성과 편리성을 제공하는 온라인 기반 심리 상담 서비스가 주목받고 있습니다.
특히 AI와 대규모 언어 모델(LLM)의 발전으로, 얼굴 표정, 음성, 시선 추적 등을 활용한 정밀한 감정 분석 기술이 가능해지면서 개인 맞춤형 상담 제공이 현실화되고 있습니다. 이 기술은 사용자의 감정 상태를 실시간으로 평가하고 적합한 조언과 지원을 제공할 수 있는 잠재력을 가지고 있습니다.
AI를 활용한 심리 상담 웹사이트는 시간과 비용의 부담을 줄이고, 더 많은 사람들이 심리적 지원을 받을 수 있도록 돕는 중요한 도구로 자리 잡을 것입니다.

## 📖 About
본 프로젝트의 개발목표는 인공지능과 대규모 언어 모델 (LLM)을 활용하여 사용자의 감정 상태를 다각도로 분석하고, 이를 바탕으로 맞춤형 심리상담 서비스를 제공하는 웹사이트를 개발하는 것입니다. 사용자는 다양한 상담 목표 중에서 자신의 필요에 맞는 목표를 선택하여 설정할 수 있습니다. 이 웹사이트의 주요 기능은 얼굴 표정 인식, 아이트래킹(시선 추적), 눈 깜빡임 검출, 음성 인식 등을 통해 사용자의 감정 상태를 정밀하게 분석하며, 이 데이터를 바탕으로 심리상담과 관련된 통계 및 분석 기능을 제공합니다.

## 🔎 Features
1. 얼굴 표정 인식을 통한 감정 분석으로 웹캠을 통해 사용자의 얼굴 표정을 실시간으로 인식
2. 아이트래킹 기능을 통해 사용자의 시선 움직임을 추적하여, 주의 집중 상태와 관심 대상 등을 파악
3. 눈 깜빡임 빈도를 분석하여 피로도, 스트레스 수준 등을 평가
4. 음성 인식 기술을 사용하여 사용자의 목소리에서 감정을 분석
5. 사용자가 제공하는 상담 내용, 감정 데이터, 배경 설명서 등을 LLM(대규모 언어 모델)을 통해 분석

## 🗣 Service
### [학습 시킬 데이터 Augmentation과 전처리]
![image](https://github.com/user-attachments/assets/66943c09-80e1-41e7-8060-aa11eee99fde)
- 웹캠을 통해 사용자의 얼굴 표정을 실시간으로 인식하고, AI 모델을 활용해 사용자의 감정 상태를 분석합니다. 얼굴 표정을 분석하여 기쁨, 슬픔, 분노, 놀람, 공포 등 다양한 감정 상태를 식별합니다.

사용된 모델 : mini_XCEPTION

 ----------------------------------------------------------------------------
  
### [얼굴 표정 눈 깜빡임 인식]
![image](https://github.com/user-attachments/assets/447b2e80-3c60-41b1-a835-c695b6e8a5a9)
- 아이트래킹 기술을 통해 사용자의 시선 움직임을 추적하여, 주의 집중 상태와 관심 대상 등을 파악합니다.
- 음성 인식 기술을 사용하여 사용자의 목소리에서 감정을 분석합니다. 목소리의 톤, 속도, 음량 등을 분석하여 기쁨, 슬픔, 분노, 불안 등 감정 상태를 식별합니다. 음성 데이터는 텍스트로 변환되어 사용자의 발언 내용과 함께 분석에 활용됩니다.

사용된 모델 : HubertForSequenceClassification

![image](https://github.com/user-attachments/assets/b8bb150a-6f71-4b95-b379-e9c759454719)
![image](https://github.com/user-attachments/assets/21a9a0d6-017f-4146-b316-27684022f4de)

------------------------------------------------------------------------
### [최종 분석]
![image](https://github.com/user-attachments/assets/16d7c21b-87b6-4c4d-806d-f7d9d83e485d)
![image](https://github.com/user-attachments/assets/9fa2f45a-f91b-4721-babf-85617bd34814)
- LLM을 이용하여 사용자가 제공하는 상담 내용, 감정 데이터, 배경 설명서 등을 분석합니다. 
- 눈을 자주 깜박일 경우 불안으로 간주되며, 눈 깜박임 빈도를 분석하여 피로도, 스트레스 수준 등을 평가합니다.

사용된 모델 : CNNModel
얼굴 표정, 아이트래킹, 음성 인식, LLM 분석 결과를 종합하여 사용자의 감정 상태에 맞춘 상담 콘텐츠를 제공합니다. 

## Team
감자빵먹고싶다

|Name|Department|Contact|
|---|---|---|
| Kang Ji Hyun | Major of Bigdata | kangjihyunlo@naver.com|
| Hwang Seo Yeon | Major of Bigdata |cindyand1q2@gmail.com|

## Install

```python
pip install opencv-python
pip install opencv-python-headless
pip install tensorflow
pip install keras
pip install soundfile
pip install sounddevice wavio
pip install openai
pip install Flask
conda install -c conda-forge pytorch torchvision torchaudio cudatoolkit=11.8  
```

## 🛠 Tool
![image](https://github.com/user-attachments/assets/1a1dae06-c978-4d20-ad2e-934635a9189e)
![image](https://github.com/user-attachments/assets/bec72076-427c-47ed-bb08-6d184ea36846)
![image](https://github.com/user-attachments/assets/f04dc29a-c335-41a2-b12b-bef9a8c802e8)





