import tqdm
import numpy as np
import pandas as pd
# Encoders for transforming text into numerical vectors
import models.Encoders.Word2Vec as Word2Vecencoder
import models.Encoders.TFIDF as TFIDFencoder
# Rankers for determining the importance of sentences 
import models.Rankers.LogisticRegression as LogisticRegressionranker
import models.Rankers.LinearRegression as LinearRegressionranker
# Decoders for converting the output of rankers into summaries
import models.Decoders.PureScore as PureScore
import models.Decoders.PureScore as PureRougeSearch
import models.Decoders.PureScore as GuidedRougeSearch

class ExtractiveSummarizer:
    # Define hyperparameter search space for different components
    hyperparameter_search_space = {'tfidf' : {'vector_size': [100,200,300,400,500,600,1000] }, 
                            'word2vec': {'vector_size': [100,200,300] , 'window': [3,4,5,6,7,8]},
                            'Logistic Regression': {'num_steps': [100,1000,10000], 'learning_rate': [0.1, 0.2, 0.3]},
                            'PureRougeSearch' : { 'beam_size': [5,10,15,20]},
                            'GuidedRougeSearch' : { 'beam_size': [5,10,15,20]},
                            'k': [2,3,4,5] }
    
    # Initialize the summarizer with specified encoder, ranker, and decoder
    def __init__(self, encoder=TFIDFencoder, ranker=LogisticRegressionranker, decoder=GuidedRougeSearch):
        self.encoder = encoder
        self.ranker = ranker
        self.decoder = decoder
        
        # Set default hyperparameters
        self.hyperparameters = {'tfidf' : {'vector_size': 300 }, 
                                'word2vec': {'vector_size': 300 , 'window': 5},
                                'Logistic Regression': {'num_steps': 1000, 'learning_rate': 0.1},
                                'PureRougeSearch' : { 'beam_size': 10}, # Note that the beam size is assumed to be larger than k 
                                'GuidedRougeSearch' : { 'beam_size': 10}, # Note that the beam size is assumed to be larger than k 
                                'k': 3 }
        # Initialize variables to store the model's state, statistics, and outputs
        self.word_vectors = None # stores the matrix where each row is a word in the vocabulary
        self.vocabulary_to_rows = None # the mapping from words to the rows of self.word_vectors
        self.normalization_mean = None # the mean used for normalization during training time is stored to be used during prediction
        self.normalization_std = None # the standard deviation used for normalization during training time is stored to be used during prediction
        self.W = None # Ranker's weights
        self.loss_list = None # loss list for analyzing logit
        self.logit_stats = None # logit statistics for analyzing logit

    def preprocess(self, X):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        
        split_articles = [[s.strip() for s in x.split('.')] for x in X]
        return split_articles
    
    def train(self, X, y):
        """
        X: list of list of sentences (i.e., comprising an article)
        y: list of yes/no decision for each sentence (as boolean)
        """

        for article, decisions in tqdm.tqdm(zip(X, y), desc="Validating data shape", total=len(X)):
            assert len(article) == len(decisions), "Article and decisions must have the same length"

        # Encoding
        # Calcualte the word vectors
        self.word_vectors , self.vocabulary_to_rows = self.encoder.construct_word_vectors(X, self.hyperparameters) # returns None
        # Calculate augmented sentence vectors for the training data
        X_vectorized = self.encoder.X_2_VectorList(X, self.word_vectors, self.vocabulary_to_rows)
        # Record normalization values
        self.normalization_mean = X_vectorized.mean(axis=0)
        self.normalization_std = X_vectorized.std(axis=0)
        # Normalize the training data
        X_vectorized = (X_vectorized - self.normalization_mean) / self.normalization_std # optional normalization
        y_flat = np.array([sentence_y for article_y_list in y for sentence_y in article_y_list ])
        # Randomly shuffle the training data
        merge = np.c_[X_vectorized,y_flat]
        np.random.shuffle(merge)
        X_vectorized = merge[:,:-1]
        y_flat = merge[:,-1]
        # Ranking: Train the ranker
        self.W , self.loss_list, self.logit_stats = self.ranker.train_ranker(X_vectorized, y_flat, self.hyperparameters)
        print("Training Completed")
        
    def predict(self, X, k=3):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        k = self.hyperparameters['k']
        for article in tqdm.tqdm(X, desc="Running extractive summarizer"):
            # Convert article into sentence vectors
            article_vectorized = self.encoder.Article_2_VectorList(article, self.word_vectors, self.vocabulary_to_rows)
            # Normalize the sentence vectors based on the mean and std of the features in the training data
            article_vectorized = (article_vectorized - self.normalization_mean) / self.normalization_std 
            # Predict scores for each sentence using the model learned by the ranker
            scores_in = self.ranker.predict_scores(article_vectorized , self.W)
            # Call the decoder to get a summary based on the scores
            sentence_scores = self.decoder.decode_scores(scores_in, article, self.hyperparameters)
            # Pick k sentences as summary
            top_k_idxs = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:k]
            top_sentences = [article[i] for i in top_k_idxs]
            summary = ' . '.join(top_sentences)
            yield summary

    # run_extractive_summarization.py adapted as a class function
    def run_extractive_summarization(self , train_data , eval_data):
        # Preprocessing for Training
        train_articles = [article['article'] for article in train_data]
        train_highligt_decisions = [article['greedy_n_best_indices'] for article in train_data] # Y
        preprocessed_train_articles = self.preprocess(train_articles) # X - divide article into sentences
        # Training
        self.train(preprocessed_train_articles, train_highligt_decisions)
        # Preprocessing for Prediction
        eval_articles = [article['article'] for article in eval_data]
        preprocessed_eval_articles = self.preprocess(eval_articles)
        # Prediction
        summaries = self.predict(preprocessed_eval_articles)
        eval_out_data = [{'article': article, 'summary': summary} for article, summary in zip(eval_articles, summaries)]
        return eval_out_data
    
    # eval.py adapted as a class function
    def eval(self, evaluator, eval_data , pred_data):
        assert len(eval_data) == len(pred_data)
        # Additional check that the articles are the same
        assert sum([eval_data[index]['article'] != pred_data[index]['article'] for index in range(len(eval_data))]) == 0 
        pred_sums = []
        eval_sums = []
        for eval, pred in tqdm.tqdm(zip(eval_data, pred_data), total=len(eval_data)):
            pred_sums.append(pred['summary'])
            eval_sums.append(eval['summary'])
        scores = evaluator.batch_score(pred_sums, eval_sums)
        # Construct a DataFrame for better display
        scores_df = pd.DataFrame(scores).transpose().rename(columns={"r" : "Recall" ,"p" : "Precision" ,"f" : "F1-score" })
        return scores_df

    # Hyperparameter selection: trains the model on train_data ad evaluates the perfromance on validation_data
    # Selects the hyperparameters that achieve highest total F-1 score for Rouge-1, Rouge-2, Rouge-4, and Rouge-L
    # The Rouge scores are computes as the average of multiple experiements
    # The number of experiments are specified by num_experiments parameter and the default value is 5
    def hyperparemeter_selection(self,  evaluator, train_data, validation_data , hyperparameters_list, num_experiments=5):

        # Initialize a dictionary to keep track of the evaluation reports for each set of hyperparameters
        hyperparameter_selection_report = {}
        # Initialize a list to keep the cumulative F1-scores for each set of hyperparameters across experiments
        hyperparam_scores = []

        # Iterate over each combination of hyperparameters provided in the list
        for hyperparams in hyperparameters_list:
            # Assign the current set of hyperparameters to be used by the model
            self.hyperparameters = hyperparameters_list
            # Run a specified number of experiments with the current set of hyperparameters
            for i in range(num_experiments):
                print('| Hyperparameters:', hyperparams, '| Experiment Number:', i, '|')
                pred_data = self.run_extractive_summarization(train_data , validation_data)
                try:
                    hyperparameter_selection_report[str(hyperparams)] += self.eval(evaluator, validation_data , pred_data) / num_experiments
                except:
                    hyperparameter_selection_report[str(hyperparams)] = self.eval(evaluator, validation_data , pred_data) / num_experiments
            score = hyperparameter_selection_report[str(hyperparams)]["F1-score"].sum()
            hyperparam_scores.append(score)

        # Find the best hyperparameters and set it as model hyperparameters
        best_hyperparameters_index = hyperparam_scores.index(max(hyperparam_scores))
        best_hyperparameters = hyperparameters_list[best_hyperparameters_index]
        self.hyperparameters = best_hyperparameters
        return pd.concat(hyperparameter_selection_report)