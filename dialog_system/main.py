from data.data_processor import train_data
from dialog_system.dialog_manager import DialogManager
from classifiers.logistic_regression import LogisticRegressionModel
from dialog_system.config import create_config_parser, Config
from dialog_system.reasoning import handle_reasoning


if __name__ == "__main__":
    parser = create_config_parser()
    args = parser.parse_args()

    config = Config(
        caps_lock=args.caps_lock,
        typo_check=args.typo_check,
        levenshtein=args.levenshtein,
        system_delay=args.system_delay,
        debug_mode=args.debug_mode,
    )

    # manager = DialogManager(FeedForwardNN(train_data, debug=DEBUG_MODE))
    manager = DialogManager(LogisticRegressionModel(train_data), config)
    suggestions = manager.converse()

    if suggestions is not None:
        handle_reasoning(suggestions)
