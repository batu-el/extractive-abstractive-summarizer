import tqdm
import random
import pandas as pd

class ExtractiveSummarizer:

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

        """
        TODO: Implement me!
        """


    def predict(self, X, k=3):
        """
        X: list of list of sentences (i.e., comprising an article)
        """
        
        for article in tqdm.tqdm(X, desc="Running extractive summarizer"):
            """
            TODO: Implement me!
            """

            # Randomly assign a score to each sentence. 
            # This is just a placeholder for your actual model.
            sentence_scores = [random.random() for _ in article]

            # Pick the top k sentences as summary.
            # Note that this is just one option for choosing sentences.
            top_k_idxs = sorted(range(len(sentence_scores)), key=lambda i: sentence_scores[i], reverse=True)[:k]
            top_sentences = [article[i] for i in top_k_idxs]
            summary = ' . '.join(top_sentences)
            
            yield summary
            
    def run_extractive_summarization(self , train_data , eval_data):
        """
        train_data: list of list of sentences (i.e., comprising an article)
        """
        # Preprocessing
        train_articles = [article['article'] for article in train_data]
        train_highligt_decisions = [article['greedy_n_best_indices'] for article in train_data] # Y
        preprocessed_train_articles = self.preprocess(train_articles) # X - divide article into sentences
        # Training
        self.train(preprocessed_train_articles, train_highligt_decisions)
        eval_articles = [article['article'] for article in eval_data]
        preprocessed_eval_articles = self.preprocess(eval_articles)
        summaries = self.predict(preprocessed_eval_articles)
        eval_out_data = [{'article': article, 'summary': summary} for article, summary in zip(eval_articles, summaries)]
        return eval_out_data

    def eval(self, evaluator, eval_data , pred_data):
        assert len(eval_data) == len(pred_data)
        assert sum([eval_data[index]['article'] != pred_data[index]['article'] for index in range(len(eval_data))]) == 0 # additional check that the articles are the same

        pred_sums = []
        eval_sums = []
        for eval, pred in tqdm.tqdm(zip(eval_data, pred_data), total=len(eval_data)):
            pred_sums.append(pred['summary'])
            eval_sums.append(eval['summary'])

        scores = evaluator.batch_score(pred_sums, eval_sums)
        scores_df = pd.DataFrame(scores).transpose().rename(columns={"r" : "Recall" ,"p" : "Precision" ,"f" : "F1-score" })
        return scores_df