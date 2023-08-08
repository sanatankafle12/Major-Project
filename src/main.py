import streamlit as st
from preprocessing import preprocess_text

st.title("For Mid-Term")

youtube_link = st.text_area("Youtube URL: ",height=10)
text = st.text_area('Text')

processed_text = preprocess_text(text)

if st.button('Generate notes and MCQ'):
    st.write(processed_text)

