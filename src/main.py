import streamlit as st
from preprocessing import preprocess_text
from model import *
import matplotlib.pyplot as plt

st.title("For Mid-Term")
text = st.text_area('Text')
sentence = text.split('.')
processed_text = preprocess_text(text)
stopWords = ['ever', 'under', 'although', 'eight', 'many', 'toward', 'would', 'thru', 'her', 'thereby', 'in', 'meanwhile', 'per', 'seeming', 'whereupon', 'anywhere', 'empty', 'then', 'there', 'here', 'twelve', 'my', 'nowhere', 'some', 'ourselves', '‘ll', 'itself', 'only', 'seemed', 'these', 'such', 'much', 'less', 'ten', 'hence', 'this', 'as', 'also', 'wherever', 'while', 'done', 'moreover', 'three', 'than', 'becomes', 'of', 'yourself', 'were', 'nothing', 'an', 'nor', 'enough', 'his', '’re', 'does', 'they', 'even', 'behind', 'may', 'take', 'afterwards', 'have', 'for', 'formerly', 'something', 'now', 'put', 'ours', 'eleven', 'none', 'out', 'besides', 'again', 'hers', 'first', 'via', 'anyhow', 'latter', 'its', 'whereby', 'hundred', 'say', 'hereby', 'not', 'with', 'often', 'a', 'before', 'but', 'each', 'becoming', 'full', 'from', 'within', 'both', 'below', 'others', 'show', 'whenever', 'too', 'mostly', 'anyway', 'mine', 'once', 'yourselves', 'hereafter', 'another', 'is', 'serious', 'few', 'together', 'might', 'go', 'n’t', 'into', 'whole', 'keep', 'thereafter', 'to', 'whither', 'how', 'further', 'otherwise', '’ll', 'due', 'fifteen', 'whether', 'sixty', 'always', 'amount', 'without', 'where', 'myself', 'who', 'using', 'by', 'made', 'should', 'what', 'nine', 'must', 'indeed', 'being', 'do', 'almost', 'up', 'hereupon', 'namely', 'however', 'amongst', 'it', 'most', 'off', 'your', 'bottom', 'so', 'him', 'perhaps', "'re", 'two', 'seems', 'regarding', 'various', '‘re', 'became', 'are', 'did', 'be', 'thus', 'move', 'and', 'above', 'ca', 'i', 'across', 'all', 'part', 'throughout', 'used', 'six', 'own', 'towards', "'s", 'quite', 'noone', 'them', 'along', "'ve", 'nevertheless', 'upon', 'someone', 'third', 'whatever', 'because', 'five', 'had', 'thereupon', "'ll", 'therefore', "'m", 'beforehand', 'please', 'any', 'am', '‘d', 'several', 'cannot', 'on', '’d', 'over', '‘m', 'the', 'us', 'onto', '’m', 'make', 'twenty', 'four', 'latterly', 'next', 'other', 'through', 'when', 'whoever', 'against', 'except', 'everywhere', 'you', 'our', 'me', "'d", '’s', 'during', 'that', '‘s', 'whom', 'if', 'more', 'n‘t', 'yet', 'never', 'was', 'just', 'anyone', 'same', 'top', 'can', 'beside', 'we', 'really', 'herein', 'fifty', 'somehow', 'among', 'she', 'could', 'though', 'beyond', 'else', 'well', 'nobody', 'whence', 'neither', 'until', 'last', 'seem', 'after', 'will', 'has', 'see', 'since', 'sometimes', 'wherein', 'anything', 'least', 'down', 'no', 'whereas', 'herself', 'himself', 'whereafter', 'very', 'been', 'doing', 'between', 'alone', 'everyone', 'still', 'those', 'at', 'thence', 'therein', 'already', '’ve', 'one', 'why', 'get', 'rather', 'former', 'side', 'or', 'every', 'forty', 'he', 'around', 'everything', 'their', 'become', 've', 'which', 'name', 're', 'either', "n't", 'back', 'sometime', 'front', 'call', 'elsewhere', 'whose', 'unless', 'themselves', 'give', 'yours', 'about', 'somewhere']


if st.button('TFIDF notes'):
    summary_tfidf = tfidf(sentence, stopWords)
    st.write("tfidf : ",summary_tfidf)


if st.button('Text_rank notes'):
    text, unique_words, words, sentences = preprocess_text(text)
    summary = text_rank(text)
    st.write("Text_rank : ",summary)

if st.button('N-grams'):
    text, unique_words, words, sentences = preprocess_text(text)
    words = ' '.join(words)
    plot = is_a_relationship(words)
    st.pyplot(plot)

if st.button('Formulas'):
    formula = Formula_identification(text)
    st.write('Formulas: ',formula)

button = st.radio("Select: ",('Bar Plot: ', 'Line Plot: '))

if button == ('Line Plot: '):
    plot = line(text,stopWords,sentence)
    st.pyplot(plot)

if button == ('Bar Plot: '):
    plot = bar(text,stopWords,sentence)
    st.pyplot(plot)