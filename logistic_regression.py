from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

MAX_ITER = 10000


class LogisticRegressionModel:
    def __init__(self, train_data):
        acts = []
        sentences = []
        for tup in train_data:
            acts.append(tup[0])
            sentences.append(tup[1])

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)
        reg = LogisticRegression(max_iter=MAX_ITER)
        fit = reg.fit(X, acts)

        self.fit = fit
        self.vectorizer = vectorizer
        self.info = f"max_iter: {MAX_ITER}"

    def predict(self, sentences):
        X_test = self.vectorizer.transform(sentences)
        return self.fit.predict(X_test)


if __name__ == "__main__":
    from data import train_data, dev_data

    n = LogisticRegressionModel(train_data)

    print(n.predict(["no, i dont want spanish"]))