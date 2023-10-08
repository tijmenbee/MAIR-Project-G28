import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


@dataclass
class InferenceRule:
    rules: List[Tuple[str, str, bool]]
    satisfied_description: Optional[str] = None
    unsatisfied_description: Optional[str] = None


DEFAULT_INFERENCE_RULES = {
    'touristic': [
        InferenceRule(
            rules=[('pricerange', 'cheap', True), ('food_quality', 'good food', True)],
            satisfied_description="a cheap restaurant with good food attracts tourists",
            unsatisfied_description="a restaurant that isn't cheap or has no good food doesn't attract tourists"
        ),
        InferenceRule(
            rules=[('food', 'romanian', False)],
            unsatisfied_description="Romanian cuisine is unknown for most tourists and they prefer familiar food",
        )
    ],
    'romantic': [
        InferenceRule(
            rules=[('length_of_stay', 'long stay', True)],
            satisfied_description="spending a long time in a restaurant is romantic",
        ),
        InferenceRule(
            rules=[('crowdedness', 'quiet', True)],
            satisfied_description="a quiet restaurant is romantic",
            unsatisfied_description="a busy restaurant is not romantic"
        ),
    ],
    'children': [
        InferenceRule(
            rules=[('length_of_stay', 'long stay', False)],
            satisfied_description="spending a long time is not advised when taking children",
            unsatisfied_description="spending a short time is best when taking children"
        )
    ],
    'assigned seats': [
        InferenceRule(
            rules=[('crowdedness', 'busy', True)],
            satisfied_description="in a busy restaurant the waiter decides where you sit",
            unsatisfied_description="in a quiet restaurant the waiter often doesn't decide where you sit"
        )
    ]
}


class Reasoning:
    def __init__(self, rules: Dict[str, List[InferenceRule]] = None):
        if rules is None:
            rules = DEFAULT_INFERENCE_RULES

        self.rules = rules
        self.all_consequents = set(self.rules.keys())

    def apply_inference_rules(self, suggestion, consequent: str) -> Tuple[List[str], bool]:
        if consequent not in self.rules:
            raise ValueError(f"Unknown consequent: {consequent}")

        inference_rules = self.rules[consequent]

        final_outcome = True
        reasonings = []
        for rule_group in inference_rules:
            rules_match = all((getattr(suggestion, rule[0]) == rule[1]) == rule[2] for rule in rule_group.rules)

            if rules_match:
                if rule_group.satisfied_description:
                    reasonings.append(rule_group.satisfied_description)
            else:
                reasonings = [rule_group.unsatisfied_description]
                final_outcome = False
                break

        return reasonings, final_outcome

    def get_extra_requirements_suggestions(self, suggestions: List, consequent: str):
        for restaurant in suggestions:
            if result := self.apply_inference_rules(restaurant, consequent):
                reasonings, rule_satisfied = result
                if rule_satisfied:
                    yield restaurant, ", and".join(reasonings)

    def handle_extra_requirements(self, all_suggestions):
        consequent = ""
        while not (m := re.search(fr"({'|'.join(self.all_consequents)})", consequent)):
            consequent = input(f"Please specify your additional requirement ({', '.join(self.all_consequents)}):\n")

        consequent = m.group(1)

        result = next(self.get_extra_requirements_suggestions(all_suggestions, consequent), None)

        if result:
            result += (consequent,)

        return result
