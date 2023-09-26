import csv
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict

from data import train_data
from inform_keywords import inform_keyword_finder
from logistic_regression import LogisticRegressionModel


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

    def output_system_message(self) -> None:
        if self.system_message:
            print(self.system_message)
            self.system_message = None

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

    def can_make_suggestion(self) -> bool:
        return bool(self._pricerange) and bool(self._area) and bool(self._food)

    def try_to_make_suggestion(self, restaurants: List[Restaurant]) -> None:
        if not self.can_make_suggestion():
            self.ask_for_missing_info()
            return

        suggestions = self.calculate_suggestions(restaurants)
        if suggestions and self.current_suggestions_index < len(suggestions):  # Suggestions exist
            suggestion = suggestions[self.current_suggestions_index]
            self.system_message = f"Here's a suggestion: {suggestion}"
            self.current_suggestion = suggestion
            self.current_suggestions_index += 1
        else:  # No suggestions exist
            self.system_message = "Sorry, there's no suggestions given your requirements. Please try something else."

    def calculate_suggestions(self, restaurants: List[Restaurant]) -> List[Restaurant]:
        suggestions = []
        for r in restaurants:
            if (r.pricerange in self._pricerange and r.area in self._area and r.food in self._food and r not in
                    self._excluded_restaurants):
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
            self.system_message = "I'm sorry, I don't understand. Could you repeat that?"  # Shouldn't happen!

    def ask_for_confirmation(self) -> None:
        confirmation_str = "Please confirm the following (yes/no):\n"

        confirmation_str += f"Price range: {', '.join(self._pricerange)}\n"
        confirmation_str += f"Area: {', '.join(self._area)}\n"
        confirmation_str += f"Food: {', '.join(self._food)}\n"

        self.system_message = confirmation_str

    def update_preferences(self, extracted_preferences) -> bool:
        updated = False
        if extracted_preferences["pricerange"]:
            self.set_price_range(extracted_preferences["pricerange"])
            updated = True
        if extracted_preferences["area"]:
            self.set_area(extracted_preferences["area"])
            updated = True
        if extracted_preferences["food"]:
            self.set_food(extracted_preferences["food"])
            updated = True

        return updated


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
        act = self.act_classifier.predict([utterance])[0]

        extracted_preferences = self.extract_preferences(utterance, dialog_state.current_preference_request)

        print("act: ", act)
        print("current prefs: ", dialog_state._pricerange, dialog_state._area, dialog_state._food)
        print("extracted prefs: ", extracted_preferences)

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
                elif preferences_changed:
                    dialog_state.try_to_make_suggestion(self.all_restaurants)
                else:  # User didn't provide any new prefs - ask for them.
                    dialog_state.system_message = "Sorry for misunderstanding - please provide your preferences again."

            else:  # User denies suggestion
                if not preferences_changed:  # User didn't provide any new prefs - give next suggestion
                    dialog_state.add_excluded_restaurant(dialog_state.current_suggestion)

                dialog_state.try_to_make_suggestion(self.all_restaurants)

        if act in ["reqalts", "reqmore"]:
            preferences_changed = dialog_state.update_preferences(extracted_preferences)
            if preferences_changed and dialog_state.can_make_suggestion():
                dialog_state.ask_for_confirmation()
            else:
                dialog_state.try_to_make_suggestion(self.all_restaurants)

        print(f"new prefs: {dialog_state._pricerange=}, {dialog_state._area=}, {dialog_state._food=}")

        return dialog_state

    def converse(self):
        print("Hello! Welcome to our restaurant recommendation system!")

        dialog_state = DialogState()
        dialog_state.ask_for_missing_info()
        dialog_state.output_system_message()
        while not dialog_state.conversation_over:
            dialog_state = self.transition(dialog_state, input("> "))
            dialog_state.output_system_message()

    @staticmethod
    def extract_preferences(user_input, preference_type: PreferenceRequest) -> Dict[str, List[str]]:
        return inform_keyword_finder(user_input, preference_type.value)


if __name__ == "__main__":
    manager = DialogManager(LogisticRegressionModel(train_data))
    manager.converse()
