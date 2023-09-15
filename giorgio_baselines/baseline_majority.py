from collections import Counter
from typing import List


class BaselineMajority:
    def __init__(self, acts: List[str]):
        counts = Counter(acts)
        counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

        majority_act = counts[0][0]  # Should be "inform"
        assert majority_act == "inform"

        self.majority_act = majority_act
        self.info = "majority act: \"inform\""

    def predict(self, sentences: List[str]) -> List[str]:
        return [self.majority_act] * len(sentences)
