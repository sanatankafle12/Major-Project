import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words and word.isalpha()]
        words.extend(filtered_tokens)

    # Stemming using PorterStemmer
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in words]

    # Lemmatization using WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]

    # Get unique words
    unique_words = set(lemmatized_words)

    return (unique_words, lemmatized_words, sentences)


unique_words, words, sentences = preprocess_text("Hello")
print(unique_words)
