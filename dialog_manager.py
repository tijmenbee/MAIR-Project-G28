import csv
import itertools
from typing import List, Dict, Optional

from config import Config
from dialog_state import DialogState, Restaurant, PreferenceRequest
from inform_keywords import inform_keyword_finder, adjusted_levenshtein, request_keyword_finder


class DialogManager:
    def __init__(self, act_classifier, config: Config = None):
        self.act_classifier = act_classifier
        self.all_restaurants = []

        with open('restaurant_info.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                self.all_restaurants.append(Restaurant(*row))

        self.all_restaurants.pop(0)  # Remove header

        self.foodlist = set(r.food for r in self.all_restaurants)

        if config is None:
            config = Config()

        self.config = config

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
            dialog_state.config.update_config()
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
            dialog_state = DialogState(self.config)
            dialog_state.ask_for_missing_info()

        if dialog_state.config.debug_mode:
            print(f"new prefs: {dialog_state._pricerange=}, {dialog_state._area=}, {dialog_state._food=}")

        return dialog_state

    def converse(self) -> Optional[List[Restaurant]]:
        dialog_state = DialogState(self.config)
        dialog_state.system_message = "Hello! Welcome to our restaurant recommendation system! To change your settings, type -config at any time"
        dialog_state.output_system_message()

        dialog_state.ask_for_missing_info()
        dialog_state.output_system_message()
        while not dialog_state.conversation_over:
            user_input = input("> ").lower().strip()

            dialog_state = self.transition(dialog_state, user_input)
            dialog_state.output_system_message()

        if dialog_state.extra_requirements_suggestions:
            return dialog_state.extra_requirements_suggestions

        return None

    @staticmethod
    def extract_preferences(user_input: str, preference_type: PreferenceRequest, levenshtein_distance: int) -> Dict[str, List[str]]:
        return inform_keyword_finder(user_input, preference_type.value, levenshtein_distance)

    @staticmethod
    def extract_restaurant_info(user_input: str, levenshtein_distance: int):
        return request_keyword_finder(user_input, levenshtein_distance)
