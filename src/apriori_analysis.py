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


def main():
    col, data = analysis.load_clean_accept_data("cleanedAcceptData.csv")

    filtered_data = get_col(col, data, [0, 8, 11, 16])

    # just to test
    print("First 10 rows of filtered data:\n")
    i = 0
    for i in range(0, 10):
        print(filtered_data[i])

    print("Frequent itemsets:\n")
    # Find frequent itemsets - apriorialg.py
    L, support_data = ap.apriori(filtered_data, 0.007)

    # Generate rules - apriorialg.py
    rules = ap.generateRules(L, support_data, 0.50)
    print_rules(rules)


main()
