import re
from typing import Set

import Levenshtein
import nltk
import pandas as pd

nltk.download('words', quiet=True)

from nltk.corpus import words

words_set = set(words.words())


file = pd.read_csv("data/raw_data/restaurant_info.csv")

KEYWORDS_AREA = file["area"].unique()
KEYWORDS_AREA = [x for x in KEYWORDS_AREA if str(x) != 'nan']

KEYWORDS_PRICE = file["pricerange"].unique()
KEYWORDS_FOOD = file["food"].unique()

KEYWORDS_POSTCODE = ["postcode", "post", "postal"]
KEYWORDS_ADDRESS = ["address", "where", "location"]
KEYWORDS_PHONENUMBER = ["phone", "number", "phonenumber"]


FOOD_WORDS = ["food", "type", "cuisine"]
PRICE_WORDS = ["price", "pricerange", "money", "cost"]
AREA_WORDS = ["part", "town", "city", "location", "area"]

REGEX_ANY = [r"(doesn'?t|don'?t|does not|do not)\s?\w*?\s?(matter|care|mind)", r"\bany", r"no\spref(?:erence)?s?"]


LEVENSHTEIN_DISTANCE = 3


def request_keyword_finder(sentence: str, levenshtein_distance=LEVENSHTEIN_DISTANCE) -> Set[str]:
    request_keywords = set()
    list_keywords = {"postcode": KEYWORDS_POSTCODE, "address": KEYWORDS_ADDRESS, "phone number": KEYWORDS_PHONENUMBER}

    for word in sentence.split():
        for info, keywords in list_keywords.items():
            if any(adjusted_levenshtein(keyword, word) < levenshtein_distance for keyword in keywords):
                request_keywords.add(info)

    return request_keywords


def inform_keyword_finder(sentence: str, type=None, levenshtein_distance=LEVENSHTEIN_DISTANCE):
    area = []
    price = []
    food = []
    any = False

    inform_dict = {}

    for regex in REGEX_ANY:
        # if there is a form of 'any', e.g. any food is fine, we check for the smallest distance between a 'type' word and the 'any' word
        if re.search(regex, sentence):
            any_location = sentence.find(re.search(regex, sentence).group(0))  # The location of the 'any' type
            smallest_distance = 999  

            temp_type = None
            
            for word in FOOD_WORDS:  # If we find a food word in FOOD_WORDS we set the distance between the 'type' and 'any' to smallest distance
                if sentence.find(word) == -1:
                    continue
                if abs(sentence.find(word)-any_location) < smallest_distance:
                    smallest_distance = abs(sentence.find(word)-any_location)
                    temp_type = 'food'

            for word in AREA_WORDS:  # Ditto for area words
                if word == "center":
                    word = "centre"
                if sentence.find(word) == -1:
                    continue
                if abs(sentence.find(word)-any_location) < smallest_distance:
                    smallest_distance = abs(sentence.find(word)-any_location)
                    temp_type = 'area'

            for word in PRICE_WORDS:  # Ditto for price words
                if sentence.find(word) == -1:
                    continue
                if abs(sentence.find(word)-any_location) < smallest_distance:
                    smallest_distance = abs(sentence.find(word)-any_location)
                    temp_type = 'pricerange'

            if smallest_distance != 999 and temp_type:  # if we found a word type we set that type to 'any' in the inform_dict
                inform_dict[temp_type] = [('any', True)]
            else: 
                any = True

    for word in sentence.split():
        for keyword in KEYWORDS_FOOD:
            if adjusted_levenshtein(keyword, word) < levenshtein_distance:
                food.append((keyword, keyword == word))
                inform_dict['food'] = food
                break

        for keyword in KEYWORDS_AREA:
            if word == "center":
                word = "centre"
            if adjusted_levenshtein(keyword, word) < levenshtein_distance:
                area.append((keyword, keyword == word))
                inform_dict['area'] = area
                break

        for keyword in KEYWORDS_PRICE:
            if adjusted_levenshtein(keyword, word) < levenshtein_distance:
                price.append((keyword, keyword == word))
                inform_dict['pricerange'] = price
                break

    if any:
        inform_dict[type] = [('any', True)]

    return inform_dict


def adjusted_levenshtein(keyword: str, word: str) -> int:
    # Don't autocorrect valid English words to other words
    if word in words_set and word != keyword:
        return 10
    # Don't autocorrect words that start with different letters
    if keyword[0] != word[0]:
        return 10
    return Levenshtein.distance(keyword, word)
