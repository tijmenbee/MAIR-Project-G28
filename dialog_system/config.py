import argparse
from dataclasses import dataclass

CAPS_LOCK = False
TYPO_CHECK = False
LEVENSHTEIN_DISTANCE = 3
SYSTEM_DELAY = 0
DEBUG_MODE = False
INFORMAL = False


@dataclass
class Config:
    caps_lock: bool = CAPS_LOCK
    typo_check: bool = TYPO_CHECK
    levenshtein: int = LEVENSHTEIN_DISTANCE
    system_delay: float = SYSTEM_DELAY
    debug_mode: bool = DEBUG_MODE
    informal: bool = INFORMAL

    def update_config(self):
        quit_config = False
        while not quit_config:
            print(f"Current settings:\n"
                  f"capslock: {self.caps_lock}\n"
                  f"typochecker: {self.typo_check}\n"
                  f"levenshtein distance: {self.levenshtein}\n"
                  f"system delay: {self.system_delay} seconds\n"
                  f"debug: {self.debug_mode}\n")
            text = input("To change a setting, type \"[setting] [value]\". e.g. \"capslock True\"\nTo go back, "
                         "type \'return\':\n")
            splitinput = str(text).split()
            if splitinput[0] == "return":
                quit_config = True
            if splitinput[0] == "capslock":
                self.caps_lock = (splitinput[1].lower() == 'true')
            if splitinput[0] == "typochecker":
                self.typo_check = (splitinput[1].lower() == 'true')
            if splitinput[0] == "levenshtein":
                self.levenshtein = [int(i) for i in splitinput if i.isdigit()][0]
            if splitinput[0] == "system":
                self.system_delay = [float(i) for i in splitinput if i.replace('.', '', 1).isdigit()][0]
            if splitinput[0] == "debug":
                self.debug_mode = (splitinput[1].lower() == 'true')
            if splitinput[0] == "informal":
                self.informal = (splitinput[1].lower() == 'true')


def create_config_parser():
    parser = argparse.ArgumentParser(description="Configure system settings.")
    parser.add_argument("--capslock", type=bool, default=CAPS_LOCK, help="Enable or disable caps lock.")
    parser.add_argument("--typocheck", type=bool, default=TYPO_CHECK, help="Enable or disable typo double-checking.")
    parser.add_argument("--levenshtein", type=int, default=LEVENSHTEIN_DISTANCE, help="Set the Levenshtein distance for auto-correct.")
    parser.add_argument("--system-delay", type=float, default=SYSTEM_DELAY, help="Set the system output delay (in seconds).")
    parser.add_argument("--debug-mode", type=bool, default=DEBUG_MODE, help="Enable or disable debug mode.")
    parser.add_argument("--version0", action='store_false', help="Enable or disable informal mode (if False, "
                                                                        "system will use 'neutral' language.")

    return parser
