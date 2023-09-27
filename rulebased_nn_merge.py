import re
from typing import List

from feedforward_nn import FeedForwardNN


RULES = [  # 93.1%
    ("phone", "request"),
    ("address", "request"),
    ("sil", "null"),
    ("more", "reqmore"),
    ("code", "request"),
    ("else", "reqalts"),
    (r"\bno\b", "negate"),
    ("bye", "bye"),
    ("right", "affirm"),  # 0.2%+
    ("hello", "hello"),  # 0.2%+
    ("another", "reqalts"),  # 0.1%+
    ("where", "request"),  # 0.2%+
    ("system", "null"),  # 0.1%+
    ("repeat", "repeat"),  # 0.2%+
    ("next", "reqalts"),  # 0.1%+
]


class RuleBasedNN:
    def __init__(self, train_data, dev_data, epochs=2):
        self.rules = RULES
        self.info = f"{len(self.rules)} rules, using FeedForwardNN with {epochs} epochs"

        self.nn = FeedForwardNN(train_data, dev_data, epochs=epochs)

    def predict(self, sentences: List[str]) -> List[str]:
        labels = []
        unlabelled = []
        for i, sentence in enumerate(sentences):
            for rule, label in self.rules:
                if re.search(rule, sentence):
                    labels.append(label)
                    break
            else:
                labels.append(None)
                unlabelled.append((i, sentence))

        nn_labels = self.nn.predict([sentence for _, sentence in unlabelled])

        for (i, _), label in zip(unlabelled, nn_labels):
            labels[i] = label

        return labels


def test_model_accuracy(model, model_name: str):
    from data import deduped_dev_data
    testing_data = deduped_dev_data

    test_sentences = [sentence for act, sentence in testing_data]
    test_acts = [act for act, sentence in testing_data]

    pred_acts = model.predict(test_sentences)

    correct = sum(pred_act == test_act for pred_act, test_act in zip(pred_acts, test_acts))

    print(
        f"{model_name} accuracy: {correct / len(testing_data) * 100:.1f}%")


if __name__ == "__main__":
    from data import deduped_train_data, deduped_dev_data

    n = RuleBasedNN(deduped_train_data, deduped_dev_data, epochs=4)
    test_model_accuracy(n, "DedupedFeedForwardNN")
