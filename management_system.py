import csv
from dataclasses import dataclass
from typing import List, Optional, Dict

from baseline_rulebased import BaselineRuleBased
from data import train_data
from inform_keywords import inform_keyword_finder
from logistic_regression import LogisticRegressionModel


@dataclass
class Restaurant:
    name: str
    pricerange: str
    area: str
    food: str
    phone: str
    address: str
    postcode: str


class DialogState:
    def __init__(self):
        self._pricerange: List[str] = []
        self._area: List[str] = []
        self._food: List[str] = []
        self._excluded_restaurants: List[Restaurant] = []
        self.conversation_over = False
        self.current_suggestion: Optional[Restaurant] = None
        self.current_sugggestions_index = 0

    def set_price_range(self, pricerange: List[str]) -> None:
        self._pricerange = pricerange
        self.current_sugggestions_index = 0

    def set_area(self, area: List[str]) -> None:
        self._area = area
        self.current_sugggestions_index = 0

    def set_food(self, food: List[str]) -> None:
        self._food = food
        self.current_sugggestions_index = 0

    def add_excluded_restaurant(self, restaurant: Restaurant) -> None:
        self._excluded_restaurants.append(restaurant)
        self.current_sugggestions_index = 0

    def set_excluded_restaurants(self, excluded_restaurants: List[Restaurant]) -> None:
        self._excluded_restaurants = excluded_restaurants
        self.current_sugggestions_index = 0

    def can_make_suggestion(self) -> bool:
        return bool(self._pricerange) and bool(self._area) and bool(self._food)

    def make_suggestion(self, restaurants: List[Restaurant]) -> None:
        suggestions = self.calculate_suggestions(restaurants)
        if suggestions and self.current_sugggestions_index < len(suggestions):  # Suggestions exist
            suggestion = suggestions[self.current_sugggestions_index]
            print(f"Here's a suggestion: {suggestion}")
            self.current_suggestion = suggestion
            self.current_sugggestions_index += 1
        else:  # No suggestions exist
            print("Sorry, there's no suggestions given your requirements. Please try something else.")

    def calculate_suggestions(self, restaurants: List[Restaurant]) -> List[Restaurant]:
        suggestions = []
        for r in restaurants:
            if (r.pricerange in self._pricerange and r.area in self._area and r.food in self._food and r not in
                    self._excluded_restaurants):
                suggestions.append(r)

        return suggestions

    def ask_for_missing_info(self) -> None:
        if not self._pricerange:
            print("What is your price range (cheap, moderate, expensive, or no preference)?")
        elif not self._area:
            print("What area would you like to eat in (north, east, south, west, centre, or no preference)?")
        elif not self._food:
            print("What type of food would you like to eat (or no preference)? If you want a list of all possible food types, say 'foodlist'.")
        else:
            print("I'm sorry, I don't understand. Could you repeat that?")  # Shouldn't happen!

    def ask_for_confirmation(self) -> None:
        confirmation_str = "Please confirm the following (yes/no):\n"

        confirmation_str += f"Price range: {', '.join(self._pricerange)}\n"
        confirmation_str += f"Area: {', '.join(self._area)}\n"
        confirmation_str += f"Food: {', '.join(self._food)}\n"

        print(confirmation_str)


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

        extracted_info = self.extract_info(utterance)

        if act == "inform":
            if extracted_info["pricerange"]:
                dialog_state.set_price_range(extracted_info["pricerange"])
            if extracted_info["area"]:
                dialog_state.set_area(extracted_info["area"])
            if extracted_info["food"]:
                dialog_state.set_food(extracted_info["food"])

            if dialog_state.can_make_suggestion():  # Enough info to make a suggestion
                dialog_state.ask_for_confirmation()
            else:  # Not enough info to make a suggestion
                dialog_state.ask_for_missing_info()
        if act in ["affirm", "ack"]:
            if dialog_state.can_make_suggestion():  # Confirmation of prefs is affirmed.
                dialog_state.make_suggestion(self.all_restaurants)
            else:  # Not enough info to make a suggestion - keep asking.
                dialog_state.ask_for_missing_info()
        if act in ["negate", "deny"]:
            if not dialog_state.current_suggestion:  # Confirmation of prefs is denied.
                changed_prefs = False  # If user provided new prefs, we change them already.
                if extracted_info["pricerange"]:
                    dialog_state.set_price_range(extracted_info["pricerange"])
                    changed_prefs = True
                if extracted_info["area"]:
                    dialog_state.set_area(extracted_info["area"])
                    changed_prefs = True
                if extracted_info["food"]:
                    dialog_state.set_food(extracted_info["food"])
                    changed_prefs = True

                if changed_prefs:
                    dialog_state.make_suggestion(self.all_restaurants)
                else:  # User didn't provide any new prefs - ask for them.
                    print("Sorry for misunderstanding - please provide your preferences again.")
            else:  # User denies suggestion
                changed_prefs = False  # If user provided new prefs, we change them already.
                if extracted_info["pricerange"]:
                    dialog_state.set_price_range(extracted_info["pricerange"])
                    changed_prefs = True
                if extracted_info["area"]:
                    dialog_state.set_area(extracted_info["area"])
                    changed_prefs = True
                if extracted_info["food"]:
                    dialog_state.set_food(extracted_info["food"])
                    changed_prefs = True

                if not changed_prefs:  # User didn't provide any new prefs - give next suggestion
                    dialog_state.add_excluded_restaurant(dialog_state.current_suggestion)

                dialog_state.make_suggestion(self.all_restaurants)

        if act in ["reqalts", "reqmore"]:
            if not dialog_state.current_suggestion:
                dialog_state.ask_for_missing_info()

            if extracted_info["pricerange"]:
                dialog_state.set_price_range(extracted_info["pricerange"])
            if extracted_info["area"]:
                dialog_state.set_area(extracted_info["area"])
            if extracted_info["food"]:
                dialog_state.set_food(extracted_info["food"])

            dialog_state.make_suggestion(self.all_restaurants)

        if act == "bye":
            dialog_state.conversation_over = True
            print("Goodbye! Thank you for using the restaurant recommender system.")

        return dialog_state

    def converse(self):
        print("Hello! Ask for resatuant sb ybla bla area price tell me stuff")

        dialog_state = DialogState()
        dialog_state.ask_for_missing_info()
        while not dialog_state.conversation_over:
            dialog_state = self.transition(dialog_state, input("> "))

    def extract_info(self, user_input) -> Dict[str, List[str]]:
        return inform_keyword_finder(user_input)


if __name__ == "__main__":
    manager = DialogManager(LogisticRegressionModel(train_data))
    manager.converse()
