
from tkinter import Y
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from data import train_data
from data import test_data

def logistic_regression(train_data):
    acts = []
    sentences = []
    for tup in train_data:
        acts.append(tup[0])
        sentences.append(tup[1])

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(sentences)
    reg = LogisticRegression(max_iter = 10000)
    fit = reg.fit(X, acts)

    return fit, vectorizer

def eval(fit, test_data, vectorizer):
    sentences_test = []
    acts_test = []
    for tup in test_data:
        acts_test.append(tup[0])
        sentences_test.append(tup[1])
    X_test = vectorizer.transform(sentences_test)
    y_Pred = fit.predict(X_test)
    i = 0
    for x in range(len(y_Pred)):
        if y_Pred[x] == acts_test[x]:
            i = i+1
    return i/len(y_Pred)
    

fit, vectorizer = logistic_regression(train_data)
y = eval(fit, test_data, vectorizer)
print(y)