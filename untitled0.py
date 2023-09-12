# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:46:03 2023

@author: Group 28
"""

data = open("C:\\Users\yanni\Documents\Python\dialog_acts.dat", "r")
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

