import nltk
import math
import re
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
from preprocessing import *
from rouge import Rouge
import numpy as np


def Formula_identification(text):
    math_patterns = (r'((y|x)\s*=\s*.+),', r'if\s(.+)', r'({.+)')
    sentences = text.split('.')

    formulas = []
    for x in sentences:
        for pattern in math_patterns:
            if re.search(pattern, x):
                formulas.append(re.search(pattern, x).group(1))
                break
    return(formulas)


def n_grams(text):
    text = ' '.join(text)
    words = nltk.word_tokenize(text)
    n =3
    ngrams = list(nltk.ngrams(words, n))
    return(ngrams)
        

def tfidf(text, stopWords):
    frequency_matrix = {}
    for sent in text:
        freq_table = {}
        words=sent.split()
        for word in words:
            if word in stopWords:
                continue

            if word in freq_table:
                freq_table[word] += 1
            else:
                freq_table[word] = 1

        frequency_matrix[sent[:10]] = freq_table

    tf_matrix = {}
    for sentence, value in frequency_matrix.items():
        tf_table = {}
        sentence_word_count = len(value)
        for word, count in value.items():
            tf_table[word] = count/sentence_word_count
        tf_matrix[sentence] = tf_table

    words_in_doc = {}
    for sent,f_table in frequency_matrix.items():
        for word, count in f_table.items():
            if word in words_in_doc:
                words_in_doc[word]+=1
            else:
                words_in_doc[word] = 1

    idf_matrix = {}
    for sent, f_table in frequency_matrix.items():
        idf_table = {}
        for word in f_table.keys():
            idf_table[word] = math.log10(len(text)/float(words_in_doc[word]))
        idf_matrix[sent] =idf_table

    tf_idf_matrix = {}
    for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):
        tf_idf_table = {}
        for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                    f_table2.items()):  
            tf_idf_table[word1] = float(value1 * value2)
        tf_idf_matrix[sent1] = tf_idf_table

    sentenceValue = {}
    for sent, f_table in tf_idf_matrix.items():
        total_score_per_sentence = 0
        count_words_in_sentence = len(f_table)
        for word, score in f_table.items():
            total_score_per_sentence += score
        if count_words_in_sentence !=0:
            sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
        else:
            sentenceValue[sent]=0

    sumValues = 0
    for entry in sentenceValue:
        sumValues += sentenceValue[entry]
    average = (sumValues / len(sentenceValue))
    threshold = average
    sentence_count = 0
    summary = []

    for sentence in text:
        if sentence[:10] in sentenceValue and sentenceValue[sentence[:10]] >= (threshold):
            summary.append(sentence)
            sentence_count += 1
    return(summary)


def cosine_similarity(a, b):
    dot_product = sum(i * j for i, j in zip(a, b))
    norm_a = math.sqrt(sum(i * i for i in a))
    norm_b = math.sqrt(sum(j * j for j in b))
    return dot_product / (norm_a * norm_b)


def text_rank(text):
    sentences = text.split('. ')
    words = list(set(text.split()))
    adjacency_matrix = [[0 for x in range(len(sentences))] for y in range(len(sentences))]
    scores = [0 for x in range(len(sentences))]
    length = len(sentences)/2
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                a = [1 if word in sentences[i] else 0 for word in words]
                b = [1 if word in sentences[j] else 0 for word in words]
                similarity = cosine_similarity(a, b)
                adjacency_matrix[i][j] = similarity

    for i in range(len(sentences)):
        sum_adjacency = sum(adjacency_matrix[i])
        score = sum_adjacency / len(sentences)
        scores[i] = score
        
    sorted_sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
    summarized = []
    print(len(sorted_sentences))
    for x in range(int(length)):
        summarized.append(sorted_sentences[x][0])
    return summarized


def is_a_relationship(text):
    semantic_net = nx.Graph()
    grams = n_grams(text)
    list_ = ['is', 'is a', 'type', 'kind', 'category', 'class', 'group', 'species', 'example', 'instance', 'form', 'variety', 'model', 'variation', 'subtype', 'subcategory', 'subclass', 'subgroup', 'subspecies', 'subexample', 'subinstance', 'subform']
    nodes = ()
    edges = ()
    for x in grams:
        for j in x:
            if j in list_:
                nodes = nodes + (x[0], x[2], )
                edges = edges + ([x[0], x[2], {'relationship': 'is a'}], )
                print(edges)
    semantic_net.add_nodes_from(nodes)
    semantic_net.add_edges_from(edges)
    plt.figure()
    pos = nx.spring_layout(semantic_net)
    nx.draw_networkx(semantic_net, pos, with_labels=True, node_color="lightblue", font_size=10, node_size=1000)
    edge_labels = nx.get_edge_attributes(semantic_net, "relationship")
    nx.draw_networkx_edge_labels(semantic_net, pos, edge_labels=edge_labels)
    return plt


def evaluate(text,stopWords,sentence):
    rouge = Rouge()
    text, unique_words, words, sentences = preprocess_text(text)
    summary = text_rank(text)
    summary_tfidf = tfidf(sentence, stopWords)
    summary = '. '.join(summary)
    summary_tfidf = '. '.join(summary_tfidf)
    scores_text_rank = rouge.get_scores([summary], [text])
    scores_tfidf = rouge.get_scores([summary_tfidf], [text])
    metrics = ['rouge-1', 'rouge-2', 'rouge-l']
    text_rank_scores = scores_text_rank[0]
    tfidf_scores = scores_tfidf[0]
    text_rank_rouge1 = text_rank_scores['rouge-1']['f']
    text_rank_rouge2 = text_rank_scores['rouge-2']['f']
    text_rank_rougel = text_rank_scores['rouge-l']['f']

    tfidf_rouge1 = tfidf_scores['rouge-1']['f']
    tfidf_rouge2 = tfidf_scores['rouge-2']['f']
    tfidf_rougel = tfidf_scores['rouge-l']['f']

    plt.plot(metrics, [text_rank_rouge1, text_rank_rouge2, text_rank_rougel], label='TextRank', marker='o')
    plt.plot(metrics, [tfidf_rouge1, tfidf_rouge2, tfidf_rougel], label='TF-IDF', marker='o')
    plt.xlabel('ROUGE Metrics')
    plt.ylabel('F1-Score')
    plt.title('Comparison of ROUGE Scores: TextRank vs. TF-IDF')
    plt.legend()

    return plt

 
def part_of_relationship():
    pass

def semantic_net(text):
    pass
