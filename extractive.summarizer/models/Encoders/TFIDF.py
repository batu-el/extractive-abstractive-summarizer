''' Term Frequency - Inverse Document Frequency '''

# Import libraries
import random
import tqdm
from nltk.tokenize import wordpunct_tokenize
import numpy as np
import random
import tqdm
import numpy as np

# Construct word vectors based on TF-IDF
def construct_word_vectors(X,  hyperparameters):
    # Extract TF-IDF hyperparameters
    hyperparameters = hyperparameters['tfidf']
    # Sample a subset of data according to vector size specified in hyperparameters
    X = random.sample(X, hyperparameters['vector_size'])

    # Construct vocabulary from the training data
    vocabulary = set([])
    for article_i in tqdm.tqdm(range(len(X)) , desc="Constructing Vocabulary"):
        for sentence_i in range(len(X[article_i])):
            # Tokenize each sentence in the article and add words to the vocabulary
            for word in wordpunct_tokenize(X[article_i][sentence_i]):
                vocabulary.add(word)
    # Convert vocabulary set to a numpy array
    vocabulary = np.array(list(vocabulary))

    # Initialize term frequency (TF) matrix
    # 'rows' is a dictionary that maps the vocabulary words to their row index in the TF matrix
    rows = dict(zip(vocabulary, range(len(vocabulary)))) 
    columns = range(len(X)) # 'columns' correspond to articles in the subset of the corpus
    TF_matrix = np.zeros([len(rows), len(columns)])
    for article_i in tqdm.tqdm(range(len(X)) , desc="Constructing TF_matrix"):
        for sentence_i in range(len(X[article_i])):
            for word in wordpunct_tokenize(X[article_i][sentence_i]):
                TF_matrix[rows[word]][article_i] += 1

    # Calculate Inverse Document Frequency (IDF) vectors
    DF_vector = (TF_matrix > 0).sum(axis=1) # length is the size of the vocabulary: document frequency of each word
    IDF_vector = len(X) / DF_vector
    IDF_vector.shape = (IDF_vector.shape[0], 1)
    IDF_matrix = np.matmul(IDF_vector, np.ones([1, len(X)])) # IDF_matrix.shape should be the same as the TF Matrix
    # Apply Laplace smoothing and log-transform the TF and IDF matrices
    TF_matrix_smooth = TF_matrix + 1 # Add one to each term frequency to avoid log(0)
    Log_TF_matrix = np.log10(TF_matrix_smooth)
    Log_IDF_matrix = np.log10(IDF_matrix)
    # Elementwise multiplication of log-transformed TF and IDF matrices to get TF-IDF matrix
    TFIDF_matrix = np.multiply(Log_TF_matrix, Log_IDF_matrix) # elementwise multiplication
    # Return the TF-IDF matrix and the vocabulary-to-row index mapping
    return TFIDF_matrix , rows

def Word_2_Vector(word, word_vectors, vocabulary_to_rows):
    ''' The function handles the exception of the word not being in the vocabulary'''
    try:
        return word_vectors[vocabulary_to_rows[word]]
    except:
        # When the word in not in the vocabulary return a zero vector
        return np.zeros([word_vectors.shape[1]])

def Sentence_2_Vector(sentence, word_vectors, vocabulary_to_rows):
    ''' The function handles the exception of the sentence being empty'''
    # Check if the sentence is empty
    if wordpunct_tokenize(sentence) == []:
        # Return a zero vector for an empty sentence
        return np.zeros([word_vectors.shape[1]])
    else:
        # Calculate the vector representation of the sentence by averaging the vectors of its words

        return np.array([Word_2_Vector(word,word_vectors, vocabulary_to_rows) for word in wordpunct_tokenize(sentence)]).sum(axis=0) / len(sentence)

# Calculate cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    ''' The function handles the exception of the sentence being empty'''
    if np.dot(vector1, vector1) * np.dot(vector2, vector2) > 0:
        return np.dot(vector1, vector2) / ((np.dot(vector1, vector1) * np.dot(vector2, vector2))**0.5)
    else:
        return 0

# Convert an article into a list of vector representations for its sentences
def Article_2_VectorList(article, word_vectors, vocabulary_to_rows):
    ''' used in both training and prediciton '''
    sentence_reprs = np.array([Sentence_2_Vector(sentence , word_vectors, vocabulary_to_rows) for sentence in article])
    article_repr = sentence_reprs.sum(axis=0) 
    sentence_cos_similarities = np.array([[cosine_similarity(sentence_repr, article_repr)] for sentence_repr in sentence_reprs])
    augmented_sentence_reprs = np.array([np.concatenate([sentence_cos_similarities[i], sentence_reprs[i]]) for i in range(len(sentence_reprs))])
    return augmented_sentence_reprs

# Convert list of list of sentences X into a list of vector representations using the Article_2_VectorList function
def X_2_VectorList(X, word_vectors, vocabulary_to_rows):
    ''' usede in training '''
    X2vec = []
    for article in tqdm.tqdm(X, desc="Running X_2_VectorList"):
        sentence_reprs = Article_2_VectorList(article, word_vectors, vocabulary_to_rows)
        for sentence_repr in sentence_reprs:
            X2vec.append(sentence_repr)
    X2vec = np.array(X2vec)
    return X2vec
