''' Word 2 Vector Sentence Representation '''
# Import libraries
from gensim.models import Word2Vec
import numpy as np
import tqdm
from nltk.tokenize import wordpunct_tokenize

# Adjust the wordpunct_tokenize function to handle empty sentences by returning an empty string within a list
def wordpunct_tokenize_adj(sentence):
    if sentence == '':
        return ['']
    else:
        return wordpunct_tokenize(sentence)

# Constructs word vectors 
def construct_word_vectors(X,  hyperparameters):
    print("Constructing Word Vectors") 
    hyperparameters = hyperparameters['word2vec']
    X_flat = [wordpunct_tokenize_adj(sentence) for article in X for sentence in article]    
    return Word2Vec(sentences=X_flat, vector_size=hyperparameters['vector_size'], window=hyperparameters['window'], min_count=1 , workers=4).wv , None

# Checks if the words in an article are contained within the trained word vectors
def article_in_vocab(article , word_vectors):
    article_in_vocab = []
    for sentence in article:
        sentence_in_vocab = []
        for word in wordpunct_tokenize_adj(sentence):
            if word_vectors.__contains__(word):
                sentence_in_vocab.append(word)
        sentence_in_vocab = ' '.join(sentence_in_vocab)
        article_in_vocab.append(sentence_in_vocab)
    return article_in_vocab

# Calculate cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    if np.dot(vector1, vector1) * np.dot(vector2, vector2) > 0:
        return np.dot(vector1, vector2) / ((np.dot(vector1, vector1) * np.dot(vector2, vector2))**0.5)
    else:
        return 0

# Convert an article into a list of vector representations for its sentences
def Article_2_VectorList(article, word_vectors, vocabulary_to_rows): 
    article = article_in_vocab(article , word_vectors)
    sentences = article #assuming article comes in the form of a list of sentences
    sentences = [wordpunct_tokenize_adj(sentence) for sentence in sentences ]
    word2vec = lambda word: word_vectors[word]
    sentence2matrix = lambda sentence: np.array([word2vec(word) for word in sentence])
    sentence2vec = lambda sentence: sentence2matrix(sentence).sum(axis=0) 

    sentence_reprs = np.array([sentence2vec(sentence) for sentence in sentences])
    article_repr = sentence_reprs.sum(axis=0)

    sentence_cos_similarities = np.array([[cosine_similarity(sentence_repr, article_repr)] for sentence_repr in sentence_reprs])

    augmented_sentence_reprs = np.array([np.concatenate([sentence_cos_similarities[i], sentence_reprs[i]]) for i in range(len(sentences))])
    return augmented_sentence_reprs

# Convert list of list of sentences X into a list of vector representations using the Article_2_VectorList function
def X_2_VectorList(X, word_vectors, vocabulary_to_rows):
    X2vec = []
    for article in tqdm.tqdm(X, desc="Running X_2_VectorList"):
        augmented_sentence_reprs = Article_2_VectorList(article, word_vectors, vocabulary_to_rows)
        for augmented_sentence_repr in augmented_sentence_reprs:
            X2vec.append(augmented_sentence_repr)
    X2vec = np.array(X2vec)
    return X2vec