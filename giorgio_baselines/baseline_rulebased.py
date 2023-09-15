import re
from collections import Counter, defaultdict
from typing import List

from data import train_data

RULES = [  # >90%
    ("phone", "request"),
    ("yes", "affirm"),
    ("address", "request"),
    ("noise", "null"),
    ("sil", "null"),
    ("what", "request"),
    ("thank", "thankyou"),
    ("about", "reqalts"),
    ("code", "request"),
    ("else", "reqalts"),
    (r"\bno\b", "negate"),
    ("unintelligible", "null"),
    ("bye", "bye"),
]

LESS_RULES = [  # ~80%
    ("phone", "request"),
    ("yes", "affirm"),
    ("address", "request"),
    ("thank", "thankyou"),
    ("about", "reqalts"),
]


class BaselineRuleBased:
    def __init__(self, acts: List[str]):
        counts = Counter(acts)
        counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        majority_act = counts[0][0]  # Should be "inform"
        assert majority_act == "inform"

        self.majority_act = majority_act
        self.rules = RULES
        self.info = f"{len(self.rules)} rules"

    def predict(self, sentences: List[str]) -> List[str]:
        labels = []
        for sentence in sentences:
            for rule, label in self.rules:
                if re.search(rule, sentence):
                    labels.append(label)
                    break
            else:
                labels.append(self.majority_act)

        return labels


def look_at_common_words(n: int = 30) -> None:
    word_dict = defaultdict(Counter)

    for command, sentence in train_data:
        words = sentence.split()
        for word in words:
            word_dict[word][command] += 1

    word_dict = dict(word_dict)

    # sort by max count for each word in each command
    sorted_word_dict = {k: v for k, v in
                        sorted(word_dict.items(), key=lambda item: max(item[1].values()), reverse=True)}

    for word, info in list(sorted_word_dict.items())[:n]:
        print(f"{word:<13}{info}")


if __name__ == "__main__":
    look_at_common_words(50)
