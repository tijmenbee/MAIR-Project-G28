from data import train_data
from dialog_manager import DialogManager
from feedforward_nn import FeedForwardNN
from logistic_regression import LogisticRegressionModel
from config import DEBUG_MODE
from reasoning import handle_reasoning

if __name__ == "__main__":
    # manager = DialogManager(FeedForwardNN(train_data, debug=DEBUG_MODE))
    manager = DialogManager(LogisticRegressionModel(train_data))
    suggestions = manager.converse()

    if suggestions is not None:
        handle_reasoning(suggestions)
