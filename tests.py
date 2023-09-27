import io
import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

from data import train_data, deduped_train_data
from logistic_regression import LogisticRegressionModel
from management_system import DialogManager, DialogState, PreferenceRequest, Restaurant


# TODO. for tests i want a json(lines?) file pretty much just capturing inputs and outputs,
# possibly with internal states but not neccessarily. so we can also have a 'savechat' keyword
# at the end of a conversation. so we grow this jsonlines file to all these conversations.
# honestly mainly this, fuck a bunch of manual tests.


class TestDialogManager(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        json_path = Path("saved_convos.json")
        if not json_path.exists():
            raise FileNotFoundError("JSON file doesn't exist")

        with json_path.open('r') as f:
            cls.data = json.load(f)

    def test_transitions(self):
        for i, system_states_list in list(enumerate(self.data))[1:]:
            with self.subTest(i=i):
                states = []
                for state_dict in system_states_list:
                    state_dict['current_preference_request'] = PreferenceRequest[state_dict['current_preference_request']]
                    states.append(state_dict)

                manager = DialogManager(LogisticRegressionModel(deduped_train_data))
                dialog_state = DialogState()
                dialog_state.ask_for_missing_info()
                for system_state in states:
                    self.assertEqual(system_state["system_message"], dialog_state.system_message)
                    self.assertEqual(system_state["act"], manager.act_classifier.predict(
                        [system_state["user_input"]])[0]
                                      )
                    self.assertEqual(system_state["pricerange"], dialog_state._pricerange)
                    self.assertEqual(system_state["food"], dialog_state._food)
                    self.assertEqual(system_state["area"], dialog_state._area)
                    self.assertEqual(system_state["excluded_restaurants"], dialog_state._excluded_restaurants)
                    if system_state["current_suggestion"] is None:
                        self.assertIsNone(dialog_state.current_suggestion)
                    else:
                        self.assertEqual(Restaurant(**system_state["current_suggestion"]),
                                         dialog_state.current_suggestion)
                    self.assertEqual(system_state["current_preference_request"],
                                     dialog_state.current_preference_request)
                    dialog_state = manager.transition(dialog_state, system_state["user_input"])


if __name__ == "__main__":
    unittest.main()
