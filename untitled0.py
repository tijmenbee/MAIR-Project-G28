# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:46:03 2023

@author: Group 28
"""

data = open("C:\\Users\yanni\Documents\GitHub\MAIR-Project-G28\dialog_acts.dat", "r")
f = data.read()

def data_clean(data):

    instances = data.split("\n")
    instances = instances[:-1]
    actsents = []

    for instance in instances:
        act, sent = instance.split(" ", 1)
        instance = (act, sent)
        actsents.append(instance)
        
    return actsents
    
cleandata = data_clean(f)

def split_train_test(data, splitpercentage=0.85):
    split = int(len(data) * splitpercentage)
    data_train = data[:split]
    data_test = data[split:]
    return data_train, data_test

data_train, data_test = split_train_test(cleandata)

#print(data_test)

class classifiers():
    
    def baseline1(sent): #when using other data, this needs to be formalized using a most_frequent def to find most frequent act  
        return "inform"
        
    
    def baseline2(sent):
        if "bye" in sent:
            return "bye"
        if "no" in sent or "don't" in sent:
            return "deny"
        if "hi" in sent or "hello" in sent or "hey" in sent:
            return "hello"
        if "yes" in sent:
            return "affirm"
        if "okay" in sent or "ok" in sent or "um" in sent:
            return "ack"
        if "what is" in sent or "where is" in sent or "when is" in sent:
            return "request"
        if "restart" in sent or "start over" in sent:
            return "restart"
        if "more" in sent:
            return "reqmore"
        if "other" in sent or "alternative" in sent or "how about" in sent:
            return "reqalts"
        if "repeat" in sent:
            return "repeat"
        if "no" in sent:
            return "negate"
        if "is it" in sent:
            return "confirm"
        else:
            return "inform"
        
    def evaluate(data, classifier_name):
        correct = []
        incorrect = []
           
        for tup in data:
            if classifier_name == "baseline1":
                act = classifiers.baseline1(tup[1])
            elif classifier_name == "baseline2":
                act = classifiers.baseline2(tup[1])      
            else:
                return "classifier_name not found"
            
            if act == tup[0]:
                correct.append(tup)
            else: 
                incorrect.append(tup)
        n_correct = len(correct) / len(data) * 100
        return n_correct

x = classifiers.evaluate(data_train, "baseline1")
y = classifiers.evaluate(data_train, "baseline2")
print(x)
print(y)

               