import pandas as pd
import re
import Levenshtein
import nltk

nltk.download('words')

from nltk.corpus import words

file = pd.read_csv("restaurant_info.csv")


KEYWORDS_AREA = file["area"].unique()
KEYWORDS_AREA = [x for x in KEYWORDS_AREA if str(x) != 'nan']

KEYWORDS_PRICE = file["pricerange"].unique()
KEYWORDS_FOOD = file["food"].unique()

KEYWORDS_POSTCODE = ["postcode","post","postal"]
KEYWORDS_ADDRESS = ["address", "where", "location"]
KEYWORDS_PHONENUMBER = ["phone", "number"]


FOOD_WORDS = ["food", "type"]
PRICE_WORDS = ["price","pricerange","money"]
AREA_WORDS = ["part", "town", "city", "location"]


LEVENSHTEIN_DISTANCE = 3


REGEX_ANY = [r"(doesn'?t|don'?t|does not|do not)\s?\w*?\s?(matter|care|mind)", r"\bany"]


# todo solution for: no, I want spanish

def request_keyword_finder(sentence: str, levenshtein_distance=LEVENSHTEIN_DISTANCE):
    request_dict = {"postcode":False, "address":False, "phonenumber":False}
    list_keywords = {"postcode":KEYWORDS_POSTCODE, "address":KEYWORDS_ADDRESS, "phonenumber":KEYWORDS_PHONENUMBER}

    for word in sentence.split():
        for info, keywords in list_keywords.items():
            for keyword in keywords:
                if adjusted_levenshtein(keyword, word) < levenshtein_distance:
                    request_dict[info] = True
                    break

    return request_dict


def inform_keyword_finder(sentence: str, type = None, levenshtein_distance = LEVENSHTEIN_DISTANCE):
    area = list()
    price = list()
    food = list()
    any = False

    inform_dict = {}

    for regex in REGEX_ANY:
        if re.search(regex, sentence): # if there is a form of 'any', e.g any food is fine, we check for the smallest distance between a 'type' word and the 'any' word
            anyLocation = sentence.find(re.search(regex, sentence).group(0))  # The location of the 'any' type
            smallest_distance = 999  

            tempType = None  
            
            for word in FOOD_WORDS:  # If we find a food word in FOOD_WORDS we set the distance between the 'type' and 'any' to smallest distance
                if sentence.find(word) == -1:
                    continue
                if abs(sentence.find(word)-anyLocation) < smallest_distance:
                    smallest_distance = abs(sentence.find(word)-anyLocation)
                    tempType = 'food'

            for word in AREA_WORDS: # Ditto for area words
                if sentence.find(word) == -1:
                    continue
                if abs(sentence.find(word)-anyLocation) < smallest_distance:
                    smallest_distance = abs(sentence.find(word)-anyLocation)
                    tempType = 'area'

            for word in PRICE_WORDS: # Ditto for price words
                if sentence.find(word) == -1:
                    continue
                if abs(sentence.find(word)-anyLocation) < smallest_distance:
                    smallest_distance = abs(sentence.find(word)-anyLocation)
                    tempType = 'pricerange'

            if smallest_distance != 999 and tempType: # if we found a word type we set that type to 'any' in the inform_dict 
                inform_dict[tempType] =  [('any', True)]
            else: 
                any = True

    for word in sentence.split():
        for keyword in KEYWORDS_FOOD:
            if adjusted_levenshtein(keyword, word) < levenshtein_distance:
                food.append((keyword, keyword == word))
                #inform_dict['errorFood'] = word
                inform_dict['food'] = food
                break
    
    for word in sentence.split():
        for keyword in KEYWORDS_AREA:
            if adjusted_levenshtein(keyword, word) < levenshtein_distance:
                area.append((keyword, keyword == word))
                #inform_dict['errorArea'] = word
                inform_dict['area'] = area
                break
            
    for word in sentence.split():
        for keyword in KEYWORDS_PRICE:
            if adjusted_levenshtein(keyword, word) < levenshtein_distance:
                price.append((keyword, keyword == word))
                #inform_dict['errorPrice'] = word
                inform_dict['pricerange'] = price
                break

    if any:
        inform_dict[type] = [('any', True)]
    return inform_dict


words_set = set(words.words())


def adjusted_levenshtein(keyword: str, word: str) -> int:
    # Don't autocorrect valid English words to other words
    if word in words_set and word != keyword:
        return 10
    # Don't autocorrect words that start with different letters
    if keyword[0] != word[0]:
        return 10
    return Levenshtein.distance(keyword, word)


if __name__ == "__main__":
    test_sentence = "I dont care about the price.just give me italien food in any location"
    test_request = " Give me the adress and numbr and pstcode"
    print(inform_keyword_finder(test_sentence, "pricerange"))

    print(request_keyword_finder(test_request))