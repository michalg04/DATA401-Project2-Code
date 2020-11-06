import numpy as np
import pandas as pd
from readability import Readability
from sklearn.utils import shuffle
import datetime
import os
import math

from nltk.tokenize import sent_tokenize, word_tokenize

pos_train_files = [f for f in os.listdir('../data401/reviews/train/pos') if f.endswith('.txt')]
neg_train_files = [f for f in os.listdir('../data401/reviews/train/neg') if f.endswith('.txt')]

pos_test_files = [f for f in os.listdir('../data401/reviews/test/pos') if f.endswith('.txt')]
neg_test_files = [f for f in os.listdir('../data401/reviews/test/neg') if f.endswith('.txt')]

with open('opinion_lexicon/negative-words.txt') as f:
    negative = pd.Series(f.readlines())
    negative = [n.replace('\n','') for n in negative if not n.startswith(';')][3:]
    
with open('opinion_lexicon/positive-words.txt') as f:
    positive = pd.Series(f.readlines())
    positive = [p.replace('\n','') for p in positive if not p.startswith(';')][1:]

positive_emoticons = [
    ':-)', ':)', ':-]', ':]', ':-3', ':3', ':->', ':>', 
    '8-)', '8)', ':-}', ':}', ':o)', ':c)', ':^)', '=]', 
    '=)', ':-D', ':D', '8‑D', '8D', 'x‑D', 'xD', 'X‑D', 
    'XD', '=D', '=3', 'B^D', ':-))', ';-)', ';)', '*-)', 
    '*)', ';-]', ';]', ';^)', ':-,', ';D', '>:-)', '>:)', 
    '}:‑)', '}:)', '3:-)', '3:)', '>;)', '>:3', '>;3', 
    ":'-)", ":')"
]

negative_emoticons = [
    ':-(', ':(', ':-c', ':c', ':-<', ':<', ':-[', ':[', 
    ':-||', '>:[', ':{', ':@', '>:(', ":'‑(", ":'( ", "D-':", 
    'D:<', 'D:', 'D8', 'D;', 'D=', 'DX', ':-/', ':/', ':-.', 
    '>:\\', '>:/', ':\\', '=/', '=\\', ':L', '=L', ':S', ':-|', ':|'
] 

booster_words = [
    'extremely','really','super','a lot','very',
    'absolutely','completely''utterly','pretty',
    'quite','terribly','amazingly','wonderfully',
    'insanely','especially','particularly',
    'unusually','remarkably'
]

diminisher_words = [
    'kind of','kinda','almost','a bit','a little',
    'slightly','fairly','rather','fairly'
]

negation_words = [
    'not', 'wasn\'t', 'isn\'t', 'never',
    'no', 'weren\t', 'neither', 'nor',
    'aren\'t','ain\'t','can\'t', 
]

# --------------------------------------------
# Logistic Regression Implementation

np.random.seed(123)

def get_p(X, Beta, i):
    return 1/(1+math.exp(-1*np.matmul(X[i].T,Beta)))

def get_log_loss(Y, X, Beta):
    def get_single_loss(y, p):
        return -1*y*math.log(p) - (1-y)*math.log(1-p)
    
    loss = 0
    for i in range(len(X)):
        p = get_p(X, Beta, i)
        loss += get_single_loss(Y[i], p)   
        
    return loss
    
def fitLogistic(data, y, rate, tol = 0.01, maxiter = None):
    data = np.array(data)
    y = np.array(y)
    
    n = len(data)         # number of observations
    d = len(data[0])      # number of variables
    
    # initialize beta with all zeros
    Betas_0 = np.array([1]+[0 for i in range(d)])
    
    # add a column of all 1s to the data
    data2 = np.array([
        np.array([1] + list(d))
        for d in data
    ])    
    
    idxs = list(range(n))
    np.random.shuffle(idxs)
    
    for i in idxs:
        p = get_p(data2, Betas_0, i)
        Betas_1 = Betas_0 + rate*(y[i] - p)*data2[i]
    log_loss_1 = get_log_loss(y, data2, Betas_1)
        
    e = 1
    count = 0
    while e >= tol:
        for i in idxs:
            p = get_p(data2, Betas_1, i)
            Betas_2 = Betas_1 + rate*(y[i] - p)*data2[i]
            Betas_1 = Betas_2
        log_loss_2 = get_log_loss(y, data2, Betas_2)
        
        e = abs(log_loss_2 - log_loss_1)
        log_loss_1 = log_loss_2
        Betas_1 = Betas_2
        count += 1
        if maxiter and count == maxiter:
            break
        
    return Betas_2

def classifyLogistic(data, model):
    data = np.array(data)
    
    def predict_logistic(observation, model):
        log_odds = model[0] + sum([model[i+1]*observation[i] for i in range(len(observation))])
        prob_y_1 = math.exp(log_odds)/(1 + math.exp(log_odds))
        if prob_y_1 > 0.5:
            return 1
        else:
            return 0
        
    predictions = []
    for observation in data:
        predictions.append(predict_logistic(observation, model))
    return np.array(predictions)
    

# --------------------------------------------
# LDA Implementation

def fitLDA(data, ys):
    data = np.array(data)
    n = len(data)         # number of observations
    d = len(data[0])      # number of variables
    
    data_0 = data[ys == 0]
    data_1 = data[ys == 1]
    
    mu0 = np.array([data_0.mean(axis = 0)])
    mu1 = np.array([data_1.mean(axis = 0)])
    
    B = np.matmul((mu0 - mu1).T, mu0 - mu1)
    
    S0 = 0
    S1 = 0
    for i in range(len(data)):
        x = data[i]
        y = ys[i]
        if y == -1:
            S0 += (x - mu0).T.dot(x - mu0)
        else:
            S1 += (x - mu1).T.dot(x - mu1)
            
    S = S0 + S1
    
    eigen_values, eigen_vectors = np.linalg.eig(np.matmul(np.linalg.inv(S), B))
    max_vector = eigen_vectors[:,eigen_values.argmax()]
    return max_vector, mu0, mu1

def classifyLDA(data, model, mu0, mu1):  
    data = np.array(data)
    predictions = []
    for observation in data:
        # model is shape (1,4)
        # observation is shape (1,4)
        w_mu0 = np.matmul(model, mu0.T)
        w_mu1 = np.matmul(model, mu1.T)
        w_x = np.matmul(model, observation.T)

        # Classify to closest projected mean
        predictions.append(
            1 if abs(w_mu1 - w_x) < abs(w_mu0 - w_x) 
            else 0
        )
    return predictions


# --------------------------------------------
# SVM Implementation

def hingeLoss(w, x):
    return max(0,1 - w.dot(x))

def predictSVM(x, w):
    out = []
    for i in range(len(x)):
        prod = np.dot(x[i], w)
        out.append(0 if prod < 0 else 1)
    return np.array(out)

def fitSVM(x,y,tolerance,learning=.001, C=1):
    w = np.zeros(len(x.iloc[0]))
    errors = []
    current = 0
    while (True):
        X, Y = shuffle(x.values, [-1 if y_tmp == 0 else 1 for y_tmp in y])
        for i, x_sample in enumerate(X):
            if (Y[i]*np.dot(w.T,x_sample) <= 1):
                w = (1-learning)*w + learning*C*Y[i]*x_sample
            else:
                w = (1-learning)*w
        errors.append(w.dot(w)**2/2 - 1*sum([hingeLoss(w, tmp) for tmp in X]))
        if (len(errors) > 4 and sum([int(abs(errors[-1] - error)/abs(error) < tolerance) for error in errors[-4:-1]]) >= 2):
            return w
    return w