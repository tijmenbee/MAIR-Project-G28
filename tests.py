import io
import sys
import unittest
from unittest.mock import MagicMock
from management_system import DialogManager, DialogState


# TODO. for tests i want a json(lines?) file pretty much just capturing inputs and outputs,
# possibly with internal states but not neccessarily. so we can also have a 'savechat' keyword
# at the end of a conversation. so we grow this jsonlines file to all these conversations.
# honestly mainly this, fuck a bunch of manual tests.


class TestDialogManager(unittest.TestCase):

    def setUp(self):
        self.mock_classifier = MagicMock()
        self.dialog_manager = DialogManager(self.mock_classifier)

    def check_preferences(self, dialog_state, pricerange, area, food):
        self.assertEqual(dialog_state._pricerange, pricerange)
        self.assertEqual(dialog_state._area, area)
        self.assertEqual(dialog_state._food, food)

    def test_transition_bye(self):
        self.mock_classifier.predict.return_value = ["bye"]
        dialog_state = DialogState()
        new_state = self.dialog_manager.transition(dialog_state, "bye")
        self.assertTrue(new_state.conversation_over)

        dialog_state = DialogState()
        new_state = self.dialog_manager.transition(dialog_state, "goodbye")
        self.assertTrue(new_state.conversation_over)

    def test_transition_inform(self):
        self.mock_classifier.predict.return_value = ["inform"]
        dialog_state = DialogState()
        self.dialog_manager.transition(dialog_state, "I want cheap food.")
        self.check_preferences(dialog_state, ["cheap"], [], [])

    def test_transition_affirm(self):
        self.mock_classifier.predict.return_value = ["affirm"]
        dialog_state = DialogState()
        dialog_state.can_make_suggestion = MagicMock(return_value=True)
        dialog_state.try_to_make_suggestion = MagicMock()
        self.dialog_manager.transition(dialog_state, "yes")
        dialog_state.try_to_make_suggestion.assert_called()

    # ... more tests


if __name__ == "__main__":
    unittest.main()
