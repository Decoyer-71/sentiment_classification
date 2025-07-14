# 1. 개요
- 영화 리뷰 데이터 기반으로 '긍정', '부정' 감정을 분류하는 한국어 BERT 기반 딥러닝 모델
- 'Huggig Face Transsformer, 'datasets', 'TrainerAPI'. 'Streamlit'을 사용한 간단한 웹 인터페이스 제공

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
  git clone 
