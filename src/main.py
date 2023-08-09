import streamlit as st
from preprocessing import preprocess_text
from model import *

st.title("For Mid-Term")

youtube_link = st.text_area("Youtube URL: ",height=10)
text = st.text_area('Text')

processed_text = preprocess_text(text)

if st.button('Generate notes and MCQ'):
    text, unique_words, words, sentences = preprocess_text(text)
    ngram = n_grams(text)
    summary = text_rank(text)
    st.write("tfidf: ",summary)

