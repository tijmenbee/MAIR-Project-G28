import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from management_system import DialogManager, DialogState


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
        for i, system_states_list in enumerate(self.data):
            with self.subTest(i=i):
                deserialized_states = []
                for state_dict in system_states_list:
                    state_dict['current_preference_request'] = PreferenceRequest[state_dict['current_preference_request']]
                    state = from_dict(SystemState, state_dict)
                    deserialized_states.append(state)

                dialog_state = None  # Initialize your dialog state here

                for system_state in deserialized_states:
                    new_dialog_state = transition(dialog_state, system_state.user_input)
                    # Here, compare new_dialog_state and system_state
                    # self.assertEqual(new_dialog_state, system_state)


if __name__ == '__main__':


if __name__ == "__main__":
    unittest.main()
