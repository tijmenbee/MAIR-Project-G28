from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, train_data):
        acts = [act for act, _ in train_data]
        sentences = [sentence for _, sentence in train_data]

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(sentences)
        classifier = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=10,
                                            min_samples_leaf=1)
        classifier = classifier.fit(X, acts)

        self.classifier = classifier
        self.vectorizer = vectorizer
        self.info = f""

    def predict(self, sentences):
        x_predict = self.vectorizer.transform(sentences)
        return self.classifier.predict(x_predict)
