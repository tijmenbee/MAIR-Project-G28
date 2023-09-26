import pandas as pd
import re
import Levenshtein
import nltk

nltk.download('words')

from nltk.corpus import words

file = pd.read_csv("restaurant_info.csv")


KEYWORDS_AREA = file["area"].unique()
KEYWORDS_PRICE = file["pricerange"].unique()
KEYWORDS_FOOD = file["food"].unique()

KEYWORDS_AREA = [x for x in KEYWORDS_AREA if str(x) != 'nan']

REGEX_ANY = [r"(doesn'?t|don'?t|does not|do not)\s?\w*?\s?(matter|care)", r"\bany"]

# todo solution for: user: i dont care about the price range what about thai food
#                       speech act: inform(pricerange=any,food=thai)

# todo solution for: no, I want spanish

def inform_keyword_finder(sentence: str, type = None):
    
    area = list()
    price = list()
    food = list()
    any = False

    inform_dict = {}

    for regex in REGEX_ANY:
        if re.search(regex, sentence):
            any = True

    for keyword in KEYWORDS_FOOD:
        for word in sentence.split():
            if adjusted_Levenshtein(keyword, word) < 3:
                food.append((keyword, keyword == word))
                #inform_dict['errorFood'] = word

    for keyword in KEYWORDS_AREA:
        for word in sentence.split():
            if adjusted_Levenshtein(keyword, word) < 3:
                area.append((keyword, keyword == word))
                #inform_dict['errorArea'] = word
            

    for keyword in KEYWORDS_PRICE:
        for word in sentence.split():
            if adjusted_Levenshtein(keyword, word) < 3:
                price.append((keyword, keyword == word))
                #inform_dict['errorPrice'] = word

    inform_dict['area'] = area
    inform_dict['food'] = food
    inform_dict['pricerange'] = price
    if any:
        inform_dict[type] = ['any']
    return inform_dict


words_set = set(words.words())


def adjusted_Levenshtein(keyword, word):
    if word in words_set and word != keyword:
        return 10
    if keyword[0] != word[0]:
        return 10
    return Levenshtein.distance(keyword[1:], word[1:])


if __name__ == "__main__":
    test_sentence = "i want spanish. Also I want indien in city centre with a price chap"
    print(inform_keyword_finder(test_sentence, "pricerange"))
