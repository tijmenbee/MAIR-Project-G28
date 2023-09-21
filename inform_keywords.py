import pandas as pd
import re
import Levenshtein 

file = pd.read_csv("restaurant_info.csv")


KEYWORDS_AREA = file["area"].unique()
KEYWORDS_PRICE = file["pricerange"].unique()
KEYWORDS_FOOD = file["food"].unique()

KEYWORDS_AREA = [x for x in KEYWORDS_AREA if str(x) != 'nan']

KEYWORDS_DONTCARE = ["doesnt matter", "any", "dont care", "does not matter", "doesn't matter", "don't care", "do not care"]

# todo solution for:


def inform_keyword_finder(sentence: str, type = None):
    
    area = list()
    price = list()
    food = list()
    dontcare = False

    inform_dict = {}

    for keyword in KEYWORDS_DONTCARE:
        if re.search(keyword, sentence):
            dontcare = True     

    for keyword in KEYWORDS_FOOD:
        for word in sentence.split():
            x= Levenshtein.distance(keyword,word)
            if x < 3:
                food.append(keyword)
                #inform_dict['errorFood'] = word

    for keyword in KEYWORDS_AREA:
        for word in sentence.split():
            x= Levenshtein.distance(keyword,word)
            if keyword == 'centre':
                if x < 3:
                    area.append(keyword)
                    #inform_dict['errorArea'] = word
            else:
                if x < 2:
                    area.append(keyword)
                    #inform_dict['errorArea'] = word

    for keyword in KEYWORDS_PRICE:
        for word in sentence.split():
            x= Levenshtein.distance(keyword,word)
            if x < 3:
                price.append(keyword)
                #inform_dict['errorPrice'] = word

    inform_dict['area'] = area
    inform_dict['food'] = food
    inform_dict['pricerange'] = price
    if dontcare:
        inform_dict[type] = 'dontcare'
    return inform_dict
     

#test_sentence = "i want spanish, i dont care about price. Also I want indien in city centre with a moderate price"
#print(inform_keyword_finder(test_sentence, "pricerange"))