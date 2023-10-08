from classifiers.feedforward_nn import FeedForwardNN
from data.data_processor import train_data
from dialog_system.dialog_manager import DialogManager
from dialog_system.config import create_config_parser, Config
from dialog_system.reasoning import handle_reasoning


if __name__ == "__main__":
    parser = create_config_parser()
    args = parser.parse_args()

    config = Config(
        caps_lock=args.capslock,
        typo_check=args.typocheck,
        levenshtein=args.levenshtein,
        system_delay=args.system_delay,
        debug_mode=args.debug_mode,
    )

    manager = DialogManager(FeedForwardNN(train_data, debug=config.debug_mode), config)
    suggestions = manager.converse()

    if suggestions is not None:
        handle_reasoning(suggestions)
