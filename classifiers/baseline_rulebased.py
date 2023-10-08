import re
from collections import Counter
from typing import List


RULES_MORE = [  # 93.1%
    ("phone", "request"),
    ("address", "request"),
    ("noise", "null"),
    ("sil", "null"),
    ("thank", "thankyou"),
    ("about", "reqalts"),
    ("code", "request"),
    ("else", "reqalts"),
    (r"\bno\b", "negate"),
    ("unintelligible", "null"),
    ("bye", "bye"),
    (r"\bprice\b", "request"),  # 0.2%+
    ("there", "reqalts"),  # 0.2%+
    ("type", "request"),  # 0.2%+
    ("right", "affirm"),  # 0.2%+
    (r"\bdoes\b", "confirm"), # 0.2%+
    ("hello", "hello"),  # 0.2%+
    ("another", "reqalts"),  # 0.1%+
    ("where", "request"),  # 0.2%+
    ("system", "null"),  # 0.1%+
    ("repeat", "repeat"),  # 0.2%+
    ("next", "reqalts"),  # 0.1%+
    (r"\bye", "affirm"),
]

RULES = [  # 91.2%
    ("phone", "request"),
    (r"\bye", "affirm"),
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

RULES_LESS = [  # ~80%
    ("phone", "request"),
    (r"\bye", "affirm"),
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

        # Still use majority act for when no rules match.
        self.majority_act = majority_act
        self.rules = RULES_MORE
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
