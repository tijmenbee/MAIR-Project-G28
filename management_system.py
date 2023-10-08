import csv
import itertools
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Tuple

from data import train_data
from feedforward_nn import FeedForwardNN
from inform_keywords import inform_keyword_finder, adjusted_levenshtein, request_keyword_finder
from logistic_regression import LogisticRegressionModel

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
It is priced '{suggestion.pricerange}', in the {suggestion.area} of town. It serves {suggestion.food} food. You can ask for its address, phone number, or postcode."""

        if ask_for_additional:
            suggestion_str += ("\n(if you want to check for additional requirements (e.g. romantic, children, "
                               "touristic, assigned seats), say 'additional requirements')")

        return suggestion_str
    
    @staticmethod
    def request_str(suggestion: Restaurant, request_dict) -> str:
        unknown_string = "unknown, unfortunately"
        request_str = f"For '{suggestion.name}': "
        if request_dict.get("phonenumber"):
            request_str += f"\n- the phone number is {suggestion.phone if suggestion.phone else unknown_string}"
        if request_dict.get("address"):
            request_str += f"\n- the address is {suggestion.address if suggestion.address else unknown_string}"
        if request_dict.get("postcode"):
            request_str += f"\n- the postcode is {suggestion.postcode if suggestion.postcode else unknown_string}"

        return request_str

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
            extracted_info = self.extract_restaurant_info(utterance, dialog_state.config.levenshtein)
            if dialog_state.current_suggestion:
                dialog_state.system_message = dialog_state.request_str(dialog_state.current_suggestion, extracted_info)
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

    def get_extra_requirements_suggestions(self, suggestions: List[Restaurant], consequent: str) -> List[Tuple[Restaurant, str]]:
        matched_suggestions = []
        for restaurant in suggestions:
            if description := self.apply_inference_rules(restaurant, consequent):
                if description.rule_satisfied:
                    matched_suggestions.append((restaurant, description.reasoning))

        return matched_suggestions

    @staticmethod
    def apply_inference_rules(suggestion: Restaurant, consequent: str) -> Optional[Description]:
        if consequent == 'touristic':
            if suggestion.pricerange == 'cheap' and suggestion.food_quality == 'good food':
                return Description(True, "because a cheap restaurant with good food attracts tourists")
            if suggestion.pricerange != 'cheap' and suggestion.food_quality != 'good food':
                return Description(False, "because this restaurant usually does not attract tourists")
            if suggestion.food == 'romanian':
                return Description(False, "because Romanian cuisine is unknown for most tourists and they prefer familiar food")
        elif consequent == 'romantic':
            if suggestion.length_of_stay == 'long stay':
                return Description(True, "because spending a long time in a restaurant is romantic")
            if suggestion.crowdedness == 'busy':
                return Description(False, "because a busy restaurant is not romantic")
            elif suggestion.crowdedness == 'quiet':
                return Description(True, "a quiet restaurant is romantic")
        elif consequent == 'children':
            if suggestion.length_of_stay == 'short stay':
                return Description(False, "friendly because spending a long time is not advised when taking children")
            else:
                return Description(True, "friendly because you are able to spend a long time here")
        elif consequent == 'assigned seats':
            if suggestion.crowdedness == 'busy':
                return Description(True, "because in a busy restaurant the waiter decides where you sit")
            else:
                return Description(False, "because in a quiet restaurant the waiter doesn't have to decide where you sit")
        else:
            raise ValueError(f"Invalid consequent: {consequent}")

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
            consequent = ""
            while not (m := re.search(r"(romantic|children|touristic|assigned seats)", consequent)):
                consequent = input(
                    "Please specify your additional requirement (romantic, children, touristic, or assigned "
                    "seats):\n")

            consequent = m.group(1)

            suggestions = self.get_extra_requirements_suggestions(dialog_state.extra_requirements_suggestions, consequent)

            if not suggestions:
                print("Sorry, there are no suggestions given your additional requirements.")
            else:
                # return a random suggestion
                print(f"Here's a suggestion: "
                      f"{DialogState.suggestion_string(suggestions[0][0], ask_for_additional=False)}.\n"
                      f"It's {consequent} {suggestions[0][1]}.")

    @staticmethod
    def extract_preferences(user_input, preference_type: PreferenceRequest, levenshtein_distance) -> Dict[str, List[str]]:
        return inform_keyword_finder(user_input, preference_type.value, levenshtein_distance)
    
    @staticmethod
    def extract_restaurant_info(user_input, levenshtein_distance):
        return request_keyword_finder(user_input, levenshtein_distance)


if __name__ == "__main__":
    manager = DialogManager(FeedForwardNN(train_data, debug=DEBUG_MODE))
    manager.converse()
