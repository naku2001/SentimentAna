import streamlit as st
from utilities import predict_sentiment, interpretResult

st.title('Sentiment Analysis and Anomaly Detection System')

with st.form('Input_Form'):
    review = st.text_area('Please Enter Your Review Here')
    submit = st.form_submit_button('Analyze Review')

if submit:
    results = predict_sentiment(review, model_type='rf', model_path='models')

    st.write('##')
    st.write('### RESULTS')
    st.write('---')
    st.write(interpretResult(results), unsafe_allow_html=True)
    st.write('---')
