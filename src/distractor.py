import gensim
from sklearn.metrics.pairwise import cosine_similarity

def generate_distractors(word, model, topn=5):
    distractors = []

    try:
        # Get word embeddings for the target word
        word_vector = model[word]

        # Find most similar words based on cosine similarity
        similar_words = model.similar_by_vector(word_vector, topn=topn)

        for similar_word, similarity in similar_words:
            # Exclude the target word from the distractors
            if similar_word != word:
                distractors.append(similar_word)

    except KeyError:
        # Handle the case when the word is not found in the word embeddings model
        pass

    return distractors

# Load a pre-trained word embeddings model (e.g., Word2Vec or GloVe)
model = gensim.models.KeyedVectors.load_word2vec_format('path_to_pretrained_model.bin', binary=True)

# Call the function to generate distractors for a given word
word = "linear regression"
distractors = generate_distractors(word, model)

# Print the distractors
print(distractors)