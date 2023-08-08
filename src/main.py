import streamlit as st
from preprocessing import preprocess_text

st.title("For Mid-Term")

youtube_link = st.text_area("Youtube URL: ",height=10)
text = st.text_area('Text')

processed_text = preprocess_text(text)

if st.button('Generate notes and MCQ'):
    unique_words, words, sentences = preprocess_text(text)
    st.write("unique_words: ", unique_words)
    st.write("words: ", words)
    st.write("sentences: ", sentences)

