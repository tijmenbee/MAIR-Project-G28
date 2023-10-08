from dataclasses import dataclass

CAPS_LOCK = False
TYPO_CHECK = False
LEVENSHTEIN_DISTANCE = 3
SYSTEM_DELAY = 0
DEBUG_MODE = False


@dataclass
class Config:
    caps_lock: bool = CAPS_LOCK
    typo_check: bool = TYPO_CHECK
    levenshtein: int = LEVENSHTEIN_DISTANCE
    system_delay: int = SYSTEM_DELAY
    debug_mode: bool = DEBUG_MODE

