import streamlit as st
from inference import predict_sentiment

st.title('문장 감정 분석')
st.write('입력된 문장의 감정을 예측합니다.')

text = st.text_area('문장을 입력하세요')

if st.button('분석하기'):
    if not text.strip():
        st.warning('문장을 입력하세요')

    else :
        result = predict_sentiment(text)
        st.success(f"예측 결과 : {result}")

