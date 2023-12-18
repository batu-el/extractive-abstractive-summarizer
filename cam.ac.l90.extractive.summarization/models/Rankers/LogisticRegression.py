import numpy as np
import tqdm
import random

# Define the sigmoid function, which is the activation function used in logistic regression.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the cross-entropy loss function, which measures the performance of a classification model.
def cross_entropy_loss(h, y):
    return -(y * np.log(h) + (1 - y) * np.log(1 - h)).mean()

# Weighted version of the cross-entropy loss to handle class imbalance
def weighted_cross_entropy_loss(h, y):
    # Calculate the number of samples in each class
    n_samples_0 = np.sum(y == 0)
    n_samples_1 = np.sum(y == 1)
    total_samples = len(y)
    # Calculate class weights inversely proportional to class frequencies
    weight_for_0 = total_samples / (2 * n_samples_0)
    weight_for_1 = total_samples / (2 * n_samples_1)
    # Return the weighted loss
    return -(y * weight_for_1 * np.log(h) + (1 - y) * weight_for_0 * np.log(1 - h)).mean()

# Function to perform exact gradient descent for logistic regression
def exact_gradient_descent(X, h, y):
    return np.matmul(X.T, (h - y)) / y.shape[0]

# Function for exact gradient descent with class weights to address class imbalance
def weighted_exact_gradient_descent(X, h, y, class_weights):
    return np.matmul(X.T, (h - y)*class_weights) / y.shape[0]

# Function for performing minibatch gradient descent
def minibatch_gradient_descent(X, h, y):
    # Select a random minibatch of data to calculate the gradient
    # Reference: Jurafsky & Martin Equation 5.34
    minibatch_size = 64 
    minibatch_index = [random.randint(0,len(y)-1) for i in range(minibatch_size)]
    # Extract the minibatch samples
    X = X[minibatch_index]
    y = y[minibatch_index]
    h = h[minibatch_index]
    # Compute and return the gradient for the minibatch
    return np.matmul(X.T, (h - y)) / minibatch_size 

# Function for weighted minibatch gradient descent
def weighted_minibatch_gradient_descent(X, h, y, class_weights):
    # minibatch_gradient_descent incorporating class weights
    minibatch_size = 64
    minibatch_index = [random.randint(0,len(y)-1) for i in range(minibatch_size)]
    X = X[minibatch_index]
    y = y[minibatch_index]
    h = h[minibatch_index]
    class_weights = class_weights[minibatch_index]
    return np.matmul(X.T, (h - y)*class_weights) / minibatch_size 

# Function to calculate class weights based on class frequencies.
def calculate_class_weights(y):
    n_samples_0 = np.sum(y == 0)
    n_samples_1 = np.sum(y == 1)
    total_samples = len(y)
    # Calculate class weights inversely proportional to class frequencies
    weight_for_0 = total_samples / (2 * n_samples_0)
    weight_for_1 = total_samples / (2 * n_samples_1)
    class_weight = [weight_for_0 , weight_for_1]
    return np.where(y == 1, class_weight[1], class_weight[0])


# Function to predict scores. The score are setimates for P(y==1|X)
def predict_scores(X, weights):
    intercept = np.ones((X.shape[0], 1)) 
    X = np.concatenate((intercept, X), axis=1)
    return np.matmul(X, weights)

# Function to predict class labels based on the weights.
def predict(X, weights):
    return predict_scores(X, weights) >= 0.5

def calculate_stats(y_true, y_pred):
    # True positives (TP) : instances classified as positive by the model that are actually positive
    TP = np.sum((y_pred == 1) & (y_true == 1))
    # False positives (FP) : instances classified as positive by the model that are actually negative
    FP = np.sum((y_pred == 1) & (y_true == 0))
    # False negatives (FN) : instances classified as negative by the model that are actually positive
    TN = np.sum((y_pred == 0) & (y_true == 0))
    # False negatives (FN) : instances classified as negative by the model that are actually positive
    FN = np.sum((y_pred == 0) & (y_true == 1))
    # Calculate stats
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    # Return a dictionary with the calculated statistics
    return [{'precision': precision, 'recall': recall, 'accuracy': accuracy}, 
            {'TP':TP , 'FP':FP , 'TN':TN , 'FN':FN },
            {'(y_true == 0)': (y_true == 0).sum(), '(y_true == 1)': (y_true == 1).sum(), '(y_pred == 0)': (y_pred == 0).sum(), '(y_pred == 1)': (y_pred == 1).sum()}]

# Define a function to train a logistic regression ranker
def train_ranker(X, y, hyperparameters):
    num_steps= hyperparameters['Logistic Regression']['num_steps']
    learning_rate= hyperparameters['Logistic Regression']['learning_rate']
    ''' Function for Logistic Regression Training'''
    # concatente vector of 1s that has as many rows as X to represent the intercept term
    intercept = np.ones((X.shape[0], 1)) 
    X = np.concatenate((intercept, X), axis=1)
    # Alternative 1: Initialize weights to be 0
    weights = np.zeros(X.shape[1])
    # Alternative 2: Initialize weights to be random between 0 and 1.
    # weights = np.array([random.uniform(0, 1) for i in range(X.shape[1])])
    # Record the loss at each step of gradient descent
    cross_entropy_loss_progression = []
    # Calculate class weights to handle class imbalance
    class_weights = calculate_class_weights(y)
    
    for step in tqdm.tqdm(range(num_steps), desc="Training Logistic Regression"):
        z = np.matmul(X, weights)
        h = sigmoid(z)
        # Update the weights and record the loss
        gradient = weighted_minibatch_gradient_descent(X, h, y, class_weights)
        weights -= learning_rate * gradient
        # Append current losses to the progression list
        cross_entropy_loss_progression.append([cross_entropy_loss(h, y), weighted_cross_entropy_loss(h , y)])
    return weights , cross_entropy_loss_progression, calculate_stats(y, predict(X[:,1:], weights))