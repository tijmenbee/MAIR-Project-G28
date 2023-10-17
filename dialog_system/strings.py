import json


def parse_strings():
    with open("dialog_system/strings.json", "r") as f:
        return json.load(f)


strings = parse_strings()
