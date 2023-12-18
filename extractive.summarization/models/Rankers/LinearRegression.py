import numpy as np

# Function to predict scores using linear regression model
def predict_scores(X, weights):
    intercept = np.ones((X.shape[0], 1)) 
    X = np.concatenate((intercept, X), axis=1)
    return np.matmul(X, weights)

# Function to train a ranker using linear regression
def train_ranker(X, y, hyperparameters):
    # concatente vector of 1s that has as many rows as X to represent the intercept term
    intercept = np.ones((X.shape[0], 1)) 
    X = np.concatenate((intercept, X), axis=1)
    weights = np.matmul( np.matmul( np.linalg.inv( np.matmul( np.transpose(X) , X ) )  , np.transpose(X)) , y )
    return weights , None, None