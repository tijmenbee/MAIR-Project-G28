from dataclasses import dataclass

CAPS_LOCK = False
TYPO_CHECK = False
LEVENSHTEIN_DISTANCE = 3
DEBUG_MODE = False


@dataclass
class Config:
    def __init__(self, caps_lock=CAPS_LOCK, typo_check=TYPO_CHECK, levenshtein=LEVENSHTEIN_DISTANCE, debug_mode=DEBUG_MODE):
        self.caps_lock = caps_lock
        self.typo_check = typo_check
        self.levenshtein = levenshtein
        self.debug_mode = debug_mode
