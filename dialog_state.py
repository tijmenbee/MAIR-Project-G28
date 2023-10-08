import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set

from config import Config


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
    def __init__(self, config=None):
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

        if config is None:
            config = Config()

        self.config = config

    def output_system_message(self) -> None:
        if self.system_message:
            time.sleep(self.config.system_delay)
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
            suggestion_str += (" You can ask for its address, phone number, or postcode."
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
