import csv
import itertools
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Set

from data import train_data
from feedforward_nn import FeedForwardNN
from inform_keywords import inform_keyword_finder, adjusted_levenshtein, request_keyword_finder
from logistic_regression import LogisticRegressionModel
from reasoning import Reasoning

CAPS_LOCK = False
TYPO_CHECK = False
LEVENSHTEIN_DISTANCE = 3
DEBUG_MODE = False


@dataclass
class Config:
    def __init__(self, caps_lock=CAPS_LOCK, typo_check=TYPO_CHECK, levenshtein=LEVENSHTEIN_DISTANCE, debug_mode=DEBUG_MODE):
        self.caps_lock = caps_lock
        self.typo_check = typo_check
        self.levenshtein = levenshtein
        self.debug_mode = debug_mode


@dataclass
class Restaurant:
    name: str
    pricerange: str
    area: str
    crowdedness: str
    length_of_stay: str
    food: str
    food_quality: str
    phone: str
    address: str
    postcode: str


class PreferenceRequest(Enum):
    AREA = "area"
    PRICERANGE = "pricerange"
    FOOD = "food"
    ANY = None


class DialogState:
    def __init__(self):
        self._pricerange: List[str] = []
        self._area: List[str] = []
        self._food: List[str] = []
        self._excluded_restaurants: List[Restaurant] = []
        self.conversation_over = False
        self.current_suggestion: Optional[Restaurant] = None
        self.current_suggestions_index = 0
        self.system_message = None
        self.current_preference_request: PreferenceRequest = PreferenceRequest.ANY
        self.extra_requirements_suggestions = []
        self.previous_preferences = None
        self.confirm_typo = False
        self.previous_act = None
        self.typo_list = []
        self.config = Config()

    def output_system_message(self) -> None:
        if self.system_message:
            if self.config.caps_lock:
                print(self.system_message.upper())
            else:
                print(self.system_message)

    def set_price_range(self, pricerange: List[str]) -> None:
        self._pricerange = pricerange
        self.current_suggestions_index = 0

    def set_area(self, area: List[str]) -> None:
        self._area = area
        self.current_suggestions_index = 0

    def set_food(self, food: List[str]) -> None:
        self._food = food
        self.current_suggestions_index = 0

    def add_excluded_restaurant(self, restaurant: Restaurant) -> None:
        self._excluded_restaurants.append(restaurant)
        self.current_suggestions_index = 0

    def set_excluded_restaurants(self, excluded_restaurants: List[Restaurant]) -> None:
        self._excluded_restaurants = excluded_restaurants
        self.current_suggestions_index = 0

    @staticmethod
    def suggestion_string(suggestion: Restaurant, ask_for_additional=True) -> str:
        suggestion_str = f"""Here's a suggestion: {suggestion.name}!
It is priced '{suggestion.pricerange}', in the {suggestion.area} of town. It serves {suggestion.food} food."""

        if ask_for_additional:
            suggestion_str += ("You can ask for its address, phone number, or postcode."
                               "\n(if you want to check for additional requirements (e.g. romantic, children, "
                               "touristic, assigned seats), say 'additional requirements')")

        return suggestion_str
    
    @staticmethod
    def request_string(suggestion: Restaurant, requested_info: Set[str]) -> str:
        unknown_string = "unknown, unfortunately"
        request_string = f"For '{suggestion.name}': "

        if "phone number" in requested_info:
            request_string += f"\n- the phone number is {suggestion.phone if suggestion.phone else unknown_string}"
        if "address" in requested_info:
            request_string += f"\n- the address is {suggestion.address if suggestion.address else unknown_string}"
        if "postcode" in requested_info:
            request_string += f"\n- the postcode is {suggestion.postcode if suggestion.postcode else unknown_string}"

        return request_string

    def can_make_suggestion(self) -> bool:
        return bool(self._pricerange) and bool(self._area) and bool(self._food)

    def try_to_make_suggestion(self, restaurants: List[Restaurant]) -> None:
        if not self.can_make_suggestion():
            self.ask_for_missing_info()
            return

        suggestions = self.calculate_suggestions(restaurants)
        if suggestions and self.current_suggestions_index < len(suggestions):  # Suggestions exist
            suggestion = suggestions[self.current_suggestions_index]
            self.current_suggestion = suggestion
            self.system_message = self.suggestion_string(self.current_suggestion)
            self.current_suggestions_index += 1
        else:  # No suggestions exist
            self.system_message = "Sorry, there's no suggestions given your requirements. Please try something else."

    def calculate_suggestions(self, restaurants: List[Restaurant]) -> List[Restaurant]:
        suggestions = []
        for r in restaurants:
            if (
                    (r.pricerange in self._pricerange or "any" in self._pricerange) and
                    (r.area in self._area or "any" in self._area) and
                    (r.food in self._food or "any" in self._food) and
                    r not in self._excluded_restaurants
            ):
                suggestions.append(r)

        return suggestions

    def ask_for_missing_info(self) -> None:
        if not self._pricerange:
            self.system_message = "What is your price range (cheap, moderate, expensive, or no preference)?"
            self.current_preference_request = PreferenceRequest.PRICERANGE
        elif not self._area:
            self.system_message = "What area would you like to eat in (north, east, south, west, centre, " \
                                  "or no preference)?"
            self.current_preference_request = PreferenceRequest.AREA
        elif not self._food:
            self.system_message = "What type of food would you like to eat (or no preference)? If you want a list of " \
                                  "all possible food types, say 'foodlist'."
            self.current_preference_request = PreferenceRequest.FOOD
        else:
            self.system_message = "I'm sorry, I don't understand. Could you repeat that?"

    def ask_for_confirmation(self) -> None:
        confirmation_str = "Please confirm the following (yes/no):\n"

        confirmation_str += f"Price range: {', '.join(self._pricerange)}\n"
        confirmation_str += f"Area: {', '.join(self._area)}\n"
        confirmation_str += f"Food: {', '.join(self._food)}\n"

        self.system_message = confirmation_str

    def update_preferences(self, extracted_preferences) -> bool:
        updated = False
        if extracted_preferences.get("pricerange"):
            self.set_price_range(extracted_preferences["pricerange"])
            updated = True
        if extracted_preferences.get("area"):
            self.set_area(extracted_preferences["area"])
            updated = True
        if extracted_preferences.get("food"):
            self.set_food(extracted_preferences["food"])
            updated = True

        return updated

    def confirm_levenshtein(self) -> None:
        self.system_message = f"Did you mean the following: {' and '.join(self.typo_list)}?"
        self.typo_list = []

    def set_config(self):
        quit_config = False
        while not quit_config:
            print(f"Current settings:\n"
                  f"capslock: {str(self.config.caps_lock):<15}\n"
                  f"typochecker: {str(self.config.typo_check):<15}\n"
                  f"debug: {str(self.config.debug_mode):<15}\n"
                  f"levenshtein distance: {str(self.config.levenshtein):<15}")
            text = input("To change a setting, type \"[setting] [value]\". e.g. \"capslock True\"\n To go back, "
                         "type \'return\':\n")
            splitinput = str(text).split()
            if splitinput[0] == "return":
                quit_config = True
            if splitinput[0] == "capslock":
                self.config.caps_lock = (splitinput[1].lower() == 'true')
            if splitinput[0] == "typochecker":
                self.config.typo_check = (splitinput[1].lower() == 'true')
            if splitinput[0] == "debug":
                self.config.debug_mode = (splitinput[1].lower() == 'true')
            if splitinput[0] == "levenshtein":
                self.config.levenshtein = [int(i) for i in text.split() if i.isdigit()][0]


class Description:
    def __init__(self, rule_satisfied, reasoning):
        self.rule_satisfied = rule_satisfied
        self.reasoning = reasoning


class DialogManager:
    def __init__(self, act_classifier):
        self.act_classifier = act_classifier
        self.all_restaurants = []

        with open('restaurant_info.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.all_restaurants.append(Restaurant(*row))

        self.all_restaurants.pop(0)  # Remove header

        self.foodlist = set(r.food for r in self.all_restaurants)

    def transition(self, dialog_state: DialogState, utterance: str) -> DialogState:
        # We keep the implementation for the dialog system and the reasoning component separate. If a suggestion is
        # made, we inform the user that they can ask for additional requirements. If they do, we leave the dialog system
        # (which implements 1b), and move to the reasoning component (which implements 1c).
        if adjusted_levenshtein("additional requirements", utterance) < dialog_state.config.levenshtein:
            dialog_state.extra_requirements_suggestions = dialog_state.calculate_suggestions(self.all_restaurants)
            dialog_state.system_message = ""
            dialog_state.conversation_over = True
            return dialog_state

        if adjusted_levenshtein("foodlist", utterance) < dialog_state.config.levenshtein:
            dialog_state.system_message = (f"Here is a list of all possible food types:\n" +
                                           '\n'.join(sorted(self.foodlist)))
            return dialog_state

        if utterance == "-config":
            dialog_state.set_config()
            return dialog_state

        act = self.act_classifier.predict([utterance])[0]

        extracted_preferences = self.extract_preferences(utterance, dialog_state.current_preference_request,
                                                         dialog_state.config.levenshtein)

        if dialog_state.config.typo_check:
            # Checks if typos are spotted
            for word, is_correct in itertools.chain(*extracted_preferences.values()):
                if not is_correct:
                    dialog_state.typo_list.append(word)
            for word, is_correct in itertools.chain(*extracted_preferences.values()):
                if not is_correct:
                    dialog_state.confirm_typo = True
                    dialog_state.previous_preferences = extracted_preferences
                    dialog_state.previous_act = act
                    dialog_state.confirm_levenshtein()
                    return dialog_state
            # Checks if typo is confirmed
            if dialog_state.confirm_typo:
                if act == "affirm":
                    extracted_preferences = dialog_state.previous_preferences
                    act = dialog_state.previous_act
                    dialog_state.confirm_typo = False

        extracted_preferences = {k: [v[0] for v in value] for k, value in extracted_preferences.items()}
        if dialog_state.config.debug_mode:
            print("act: ", act)
            print("current prefs: ", dialog_state._pricerange, dialog_state._area, dialog_state._food)
            print("extracted prefs: ", extracted_preferences)

        if act == "repeat":
            dialog_state.output_system_message()

        if act == "hello":
            dialog_state.try_to_make_suggestion(self.all_restaurants)

        if act == "bye":
            dialog_state.conversation_over = True
            dialog_state.system_message = "Goodbye! Thanks for using our restaurant recommender."

        if act == "inform":
            dialog_state.update_preferences(extracted_preferences)

            if dialog_state.can_make_suggestion():  # Enough info to make a suggestion
                dialog_state.ask_for_confirmation()

            else:  # Not enough info to make a suggestion
                dialog_state.ask_for_missing_info()

        if act in ["affirm", "ack"]:
            dialog_state.try_to_make_suggestion(self.all_restaurants)

        if act in ["negate", "deny"]:
            preferences_changed = dialog_state.update_preferences(extracted_preferences)

            if not dialog_state.current_suggestion:  # Confirmation of prefs is denied.
                if not dialog_state.can_make_suggestion():
                    dialog_state.ask_for_missing_info()
                elif preferences_changed:  # Preferences changed - make new suggestion
                    dialog_state.ask_for_confirmation()
                else:  # User didn't provide any new prefs - ask for them.
                    dialog_state.system_message = "Sorry for misunderstanding - please provide your preferences again."

            else:  # User denies suggestion
                if not preferences_changed:  # User didn't provide any new prefs - give next suggestion
                    dialog_state.add_excluded_restaurant(dialog_state.current_suggestion)
                else:
                    dialog_state.ask_for_confirmation()

        if act in ["reqalts", "reqmore"]:
            preferences_changed = dialog_state.update_preferences(extracted_preferences)
            if preferences_changed and dialog_state.can_make_suggestion():
                dialog_state.ask_for_confirmation()
            else:
                dialog_state.try_to_make_suggestion(self.all_restaurants)

        if act == "thankyou":
            if dialog_state.can_make_suggestion():
                dialog_state.system_message = "You're welcome!"
                dialog_state.conversation_over = True
            else:
                dialog_state.ask_for_missing_info()

        if act == "confirm":
            dialog_state.ask_for_confirmation()

        if act == "request":
            if dialog_state.current_suggestion:
                requested_info = self.extract_restaurant_info(utterance, dialog_state.config.levenshtein)
                dialog_state.system_message = dialog_state.request_string(dialog_state.current_suggestion, requested_info)
            else:
                dialog_state.system_message = ("Sorry, I don't have a suggestion right now. Please provide more "
                                               "information about your preferences.")

        if act == "null":
            dialog_state.system_message = "Sorry, I didn't quite get that. Could you repeat yourself?"

        if act == "restart":
            dialog_state = DialogState()
            dialog_state.ask_for_missing_info()

        if dialog_state.config.debug_mode:
            print(f"new prefs: {dialog_state._pricerange=}, {dialog_state._area=}, {dialog_state._food=}")

        return dialog_state

    def converse(self):
        dialog_state = DialogState()
        dialog_state.system_message = "Hello! Welcome to our restaurant recommendation system! To change your settings, type -config at any time"
        dialog_state.output_system_message()

        dialog_state.ask_for_missing_info()
        dialog_state.output_system_message()
        while not dialog_state.conversation_over:
            user_input = input("> ").lower().strip()

            dialog_state = self.transition(dialog_state, user_input)
            dialog_state.output_system_message()

        if dialog_state.extra_requirements_suggestions:
            reasoning = Reasoning()
            extra_requirements_info = reasoning.handle_extra_requirements(dialog_state.extra_requirements_suggestions)

            if extra_requirements_info:
                suggestion, reason, consequent = extra_requirements_info

                print(f"{DialogState.suggestion_string(suggestion, ask_for_additional=False)}\n"
                      f"Its crowdedness is usually '{suggestion.crowdedness}', the usual length of stay is '"
                      f"{suggestion.length_of_stay}', and the food quality is '{suggestion.food_quality}'.\n"
                      f"It's classified as '{consequent}' because {reason}.")
            else:
                print("Sorry, there are no suggestions given your additional requirements.")

    @staticmethod
    def extract_preferences(user_input, preference_type: PreferenceRequest, levenshtein_distance) -> Dict[str, List[str]]:
        return inform_keyword_finder(user_input, preference_type.value, levenshtein_distance)

    @staticmethod
    def extract_restaurant_info(user_input, levenshtein_distance):
        return request_keyword_finder(user_input, levenshtein_distance)


if __name__ == "__main__":
    # manager = DialogManager(FeedForwardNN(train_data, debug=DEBUG_MODE))
    manager = DialogManager(LogisticRegressionModel(train_data))
    manager.converse()
