# -*- coding: utf-8 -*-
# Imports
import apriorialg as ap
import math as m
import analysis
from matplotlib import pyplot as py
from datetime import datetime
from copy import copy, deepcopy
import numpy as np
import re


def print_rules(rules):
    print('\nRules:\n')
    for rule in rules:
        antecedent = rule[0]
        conseq = rule[1]
        conf = rule[2]
        print("%s ---> %s conf: %f" % (antecedent, conseq, conf))


def lift(rules):
    # lift = P(Y | X) / P(Y)
    for rule in rules:
        antecedent = rule[0]
        conseq = rule[1]
        conf = rule[2]
        lift = conf / support_data[conseq]
        print("Lift value for %s ---> %s: %f" % (antecedent, conseq, lift))
    print('\n')


def interest(rules):
    # interest = P(X , Y) / (P(X) * P(Y))
    for rule in rules:
        antecedent = rule[0]
        conseq = rule[1]
        unioned_set = antecedent.union(conseq)
        interest = support_data[unioned_set] / (support_data[antecedent] * support_data[conseq])
        print("Interest value for %s ---> %s: %f" % (antecedent, conseq, interest))
    print('\n')


def PS(rules):
    # PS = P(X , Y) - (P(X) * P(Y))
    for rule in rules:
        antecedent = rule[0]
        conseq = rule[1]
        unioned_set = antecedent.union(conseq)
        PS = support_data[unioned_set] - (support_data[antecedent] * support_data[conseq])
        print("PS value for %s ---> %s: %f" % (antecedent, conseq, PS))
    print('\n')


def phi(rules):
    # phi-coefficient = (P(X , Y) - (P(X) * P(Y))) / sqrt(P(X) * (1 - P(X)) * P(Y) * (1-P(Y)) )
    for rule in rules:
        antecedent = rule[0]
        conseq = rule[1]
        unioned_set = antecedent.union(conseq)
        numerator = support_data[unioned_set] - (support_data[antecedent] * support_data[conseq])
        denominator = m.sqrt(support_data[antecedent] * (1 - support_data[antecedent]) * support_data[conseq] * (1 - support_data[conseq]))
        phi = numerator / denominator
        print("phi-coefficient value for %s ---> %s: %f" % (antecedent, conseq, phi))
    print('\n')

def get_col(data, extract):
    filtered_data = []
    for col in extract:
        filtered_data.append(analysis.extract_column(data, col))
    return filtered_data


col, data = analysis.load_clean_accept_data("cleanedAcceptData.csv")

filtered_columns = get_col(data, [0, 8, 11, 16])
filtered_data = np.array(filtered_columns).T

# just to test
i = 0
for i in range(0, 10):
    print(filtered_data[i])
# Find frequent itemsets - apriorialg.py
#L, support_data = ap.apriori(filtered_data, 0.3)

# Generate rules - apriorialg.py
#rules = ap.generateRules(L, support_data, 0.70)
#print_rules(rules)
