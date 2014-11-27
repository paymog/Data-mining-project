# -*- coding: utf-8 -*-
# Imports
import apriorialg as ap
import math as m
import analysis
import numpy as np


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

def get_col(col_names, data, extract):
    filtered_data = []
    for col in extract:
        row = analysis.extract_column(data, col)
        rounded = []
        if isinstance(row[0], int) or isinstance(row[0], float):  # it's an int
            for item in row:
                rounded.append(int(m.floor(item/1000.0) * 1000.0))
            row = rounded
        new_row = []
        for element in row:
            new_row.append(col_names[col] + ": " + str(element))
        filtered_data.append(new_row)
    return np.array(filtered_data).T



col, data = analysis.load_clean_accept_data("cleanedAcceptData.csv")

filtered_data = get_col(col, data, [0, 8, 11, 16])

# just to test
print("First 10 rows of filtered data:\n")
i = 0
for i in range(0, 10):
    print(filtered_data[i])

print("Frequent itemsets:\n")
# Find frequent itemsets - apriorialg.py
L, support_data = ap.apriori(filtered_data, 0.02)

# Generate rules - apriorialg.py
rules = ap.generateRules(L, support_data, 0.50)
print_rules(rules)
