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

        from strings import strings
        self.strings = strings["informal" if self.config.informal else "neutral"]["DIALOG_STATE"]

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

    def suggestion_string(self, suggestion: Restaurant, ask_for_additional=True) -> str:
        suggestion_str = self.strings["SUGGESTION_STRING"]["INITIAL"].format(suggestion=suggestion)
        if ask_for_additional:
            suggestion_str += " " + self.strings["SUGGESTION_STRING"]["ASK_ADDITIONAL_REQS"]

        return suggestion_str

    def request_string(self, suggestion: Restaurant, requested_info: Set[str]) -> str:
        unknown_string = self.strings["REQUEST_STRING"]["UNKNOWN"]
        request_string = self.strings["REQUEST_STRING"]["INITIAL"].format(suggestion=suggestion)

        if "phone number" in requested_info:
            request_string += "\n- " + self.strings['REQUEST_STRING']['PHONE_NUMBER'].format(
                phone_number=suggestion.phone if suggestion.phone else unknown_string)
        if "address" in requested_info:
            request_string += "\n- " + self.strings['REQUEST_STRING']['ADDRESS'].format(
                address=suggestion.address if suggestion.address else unknown_string)
        if "postcode" in requested_info:
            request_string += "\n- " + self.strings['REQUEST_STRING']['POSTCODE'].format(
                postcode=suggestion.postcode if suggestion.postcode else unknown_string)

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
            self.system_message = self.strings["SUGGESTION_STRING"]["NO_SUGGESTION_AVAILABLE"]

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
            self.system_message = self.strings["ASK_MISSING_INFO"]["PRICERANGE"]
            self.current_preference_request = PreferenceRequest.PRICERANGE
        elif not self._area:
            self.system_message = self.strings["ASK_MISSING_INFO"]["AREA"]
            self.current_preference_request = PreferenceRequest.AREA
        elif not self._food:
            self.system_message = self.strings["ASK_MISSING_INFO"]["FOOD"]
            self.current_preference_request = PreferenceRequest.FOOD
        else:
            self.system_message = self.strings["ASK_MISSING_INFO"]["OTHER"]

    def ask_for_confirmation(self) -> None:
        confirmation_str = self.strings["ASK_FOR_CONFIRMATION"]["INITIAL"] + "\n"

        pricerange_str = self.strings["ASK_FOR_CONFIRMATION"]["PRICERANGE"]
        if "any" in self._pricerange:
            pricerange_str = self.strings["ASK_FOR_CONFIRMATION"]["PRICERANGE_ANY"]

        area_str = self.strings["ASK_FOR_CONFIRMATION"]["AREA"]
        if "any" in self._area:
            area_str = self.strings["ASK_FOR_CONFIRMATION"]["AREA_ANY"]

        confirmation_str += pricerange_str.format(priceranges=', '.join(self._pricerange)) + "\n"
        confirmation_str += area_str.format(areas=', '.join(self._area)) + "\n"
        confirmation_str += self.strings["ASK_FOR_CONFIRMATION"]["FOOD"].format(foods=', '.join(self._food)) + "\n"

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
        self.system_message = self.strings["LEVENSHTEIN"]["CONFIRM"].format(typo_list=' and '.join(self.typo_list))
        self.typo_list = []
