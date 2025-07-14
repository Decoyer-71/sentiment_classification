# 1. 개요
> 영화 리뷰 데이터 기반으로 '긍정', '부정' 감정을 분류하는 한국어 BERT 기반 딥러닝 모델
>  - 'Huggig Face Transsformer, 'datasets', 'TrainerAPI'. 'Streamlit'을 사용한 간단한 웹 인터페이스 제공

# 2. 기술 스택
- Python 3.10+
- Hugging Face Transformers
- Datasets(NSMC - 네이버 영화리뷰 데이터셋)
- Pytorch
- Streamlit(웹 UI)
- Evaluate

# 3. 프로젝트 구조
  - app.py : Streamlit 웹앱 실행 파일
  - train.py : 모델학습 및 저장
  - inference.py : 감전분류 예측 함수 정의
  - model/ : 학습된 모델 저장 폴더
  - requirement.txt : 필요한 라이브러리 목록

# 4. 설치 및 실행방법
## 1) 저장소 클론 
  - bash
<pre>
  <code>
    git clone https://github.com/Decoyer-71/sentiment_classification.git 

    cd sentiment_classification
  </code>
</pre>

## 2) 가상환경 설치 및 패키지 설치
  - bash
<pre>
  <code>
  pip install -r requirements.txt
  </code>
</pre>

## 3) 모델 훈련
  - bash
<pre>
  <code>
  python train.py
  </code>
</pre>

## 4) 웹 앱 실행
  - bash
<pre>
  <code>
  streamlit run app.py
  </code>
</pre>
  - 실행결과 
<img width="650" height="450" alt="image" src="https://github.com/user-attachments/assets/c041b46c-d02f-40c1-a7ba-4ecbac713ced" />

     개뼈다귀라는 말은 사실 긍정적인 의미일지도 모른다.


    
# 5. 성능평가
  > 테스트셋 성능
  >  - Accuracy : 90%
  >  - loss : 0.284
  >  - runtime : 113초
  - 이번 프로젝트에서는 파라미터 미세조정 등 성능개선 보다는 텍스트 분류 모델에 대한 학습 차원으로 진행하였음.
  - 모델을 훈련하면서 최초에는 klue/roberta-large모델을 사전학습모델로 사용했으나, 파라미터가 너무 많아서 small 모델로 다시 조정(다음에는 세부 실험한 내용을 readme에 추가하도록 해야겠다.)
  - 사전학습 모델 변경 전 모델의 학습시간 단축을 위해 GPU 반정밀도 훈련, batch 조정을 통해 어느정도 학습시간 단축에 대한 테스트를 진행하였음. 
  - small 모델임에도 불구하고 정확도 90로 준수한 성능을 보임
  - 다음에 배포방식은 FastAPI 기반으로 REST API를 제공하는 것으로 연습할 예정

