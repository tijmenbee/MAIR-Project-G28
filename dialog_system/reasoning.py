import re
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from dialog_system.dialog_state import DialogState, Restaurant


@dataclass
class Rule:
    attribute: str
    value: str
    equal: bool


@dataclass
class RuleGroup:
    rules: List[Rule]
    give_as_reason: bool
    satisfied_description: Optional[str] = None
    unsatisfied_description: Optional[str] = None


DEFAULT_INFERENCE_RULES = {
    'touristic': [
        RuleGroup(
            rules=[Rule('pricerange', 'cheap', True), Rule('food_quality', 'good food', True)],
            give_as_reason=True,
            satisfied_description="a cheap restaurant with good food attracts tourists",
            unsatisfied_description="a restaurant that isn't cheap or has no good food doesn't attract tourists"
        ),
        RuleGroup(
            rules=[Rule('food', 'romanian', False)],
            give_as_reason=False,
            satisfied_description="Romanian cuisine is unknown for most tourists and they prefer familiar food",
        )
    ],
    'romantic': [
        RuleGroup(
            rules=[Rule('length_of_stay', 'long stay', True)],
            give_as_reason=True,
            satisfied_description="spending a long time in a restaurant is romantic",
        ),
        RuleGroup(
            rules=[Rule('crowdedness', 'quiet', True)],
            give_as_reason=True,
            satisfied_description="a quiet restaurant is romantic",
            unsatisfied_description="a busy restaurant is not romantic"
        ),
    ],
    'children': [
        RuleGroup(
            rules=[Rule('length_of_stay', 'long stay', False)],
            give_as_reason=True,
            satisfied_description="spending a long time is not advised when taking children",
            unsatisfied_description="spending a short time is best when taking children"
        )
    ],
    'assigned seats': [
        RuleGroup(
            rules=[Rule('crowdedness', 'busy', True)],
            give_as_reason=True,
            satisfied_description="in a busy restaurant the waiter decides where you sit",
            unsatisfied_description="in a quiet restaurant the waiter often doesn't decide where you sit"
        )
    ]
}


class Reasoning:
    def __init__(self, rules: Dict[str, List[RuleGroup]] = None):
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
            rules_match = all((getattr(suggestion, rule.attribute) == rule.value) == rule.equal for rule in rule_group.rules)

            if rules_match:
                # If a rule_group's rules all match, we can add its description as to why it's satisfied. We keep going.
                if rule_group.satisfied_description and rule_group.give_as_reason:
                    reasonings.append(rule_group.satisfied_description)
            else:
                # If one doesn't match, the group fails (as all need to match), and we give the reason why.
                if rule_group.unsatisfied_description and rule_group.give_as_reason:
                    reasonings = [rule_group.unsatisfied_description]
                final_outcome = False
                break

        return reasonings, final_outcome

    def get_extra_requirements_suggestions(self, suggestions: List, consequent: str):
        for restaurant in suggestions:
            if result := self.apply_inference_rules(restaurant, consequent):
                reasonings, rule_satisfied = result
                if rule_satisfied:
                    yield restaurant, ", and ".join(reasonings)

    def handle_extra_requirements(self, all_suggestions) -> Optional[Tuple[Restaurant, str, str]]:
        consequent = ""
        while not (m := re.search(fr"({'|'.join(self.all_consequents)})", consequent)):
            consequent = input(f"Please specify your additional requirement ({', '.join(self.all_consequents)}):\n")

        consequent = m.group(1)

        result = next(self.get_extra_requirements_suggestions(all_suggestions, consequent), None)

        if result:
            result += (consequent,)

        return result


def handle_reasoning(suggestions, config):
    reasoning = Reasoning()
    extra_requirements_info = reasoning.handle_extra_requirements(suggestions)

    if extra_requirements_info:
        suggestion, reason, consequent = extra_requirements_info

        print(f"{DialogState(config).suggestion_string(suggestion, ask_for_additional=False)}\n"
              f"Its crowdedness is usually '{suggestion.crowdedness}', the usual length of stay is '"
              f"{suggestion.length_of_stay}', and the food quality is '{suggestion.food_quality}'.\n"
              f"It's classified as '{consequent}' because {reason}.")
    else:
        print("Sorry, there are no suggestions given your additional requirements.")
