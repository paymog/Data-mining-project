import numpy as np
from matplotlib import pyplot as py
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier


def remove_accept_columns(row):
    del row[50]
    del row[48]
    del row[47]
    del row[45]
    del row[33]
    del row[29]
    del row[28]
    del row[26]
    del row[22]
    del row[19]
    del row[18]
    del row[1]
    del row[0]


def remove_reject_columns(row):
    del row[5]


def load_original_accept_data(file_name, state_dict):
    """
    Meant to read in the data file provided by lending club and clean it. There's another method below
    which is meant to read the file created after writing the cleaned version of this file
    """
    data = []
    income_status_dict = {"Verified": 0, "Not Verified": 1, "Source Verified": 2}
    home_ownership_dict = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 4, 'NONE': 5}
    loan_status_dict = {'Charged Off': 0, 'Default': 1, 'Issued': 2, 'Fully Paid': 3, 'Current': 4,
                        'Late (16-30 days)': 5, 'Late (31-120 days)': 6, 'In Grace Period': 7}

    with open(file_name) as f:
        column_labels = map(lambda x: x.strip("\" \n"), f.readline().split("\",\""))
        remove_accept_columns(column_labels)

        for line in f:
            d = map(lambda x: x.strip("\" \n"), line.split("\",\""))

            d[:5] = map(int, d[:5])  # cast first 5 columns to ints
            d[5] = int(d[5][:2])
            d[6] = float(d[6][:-1])
            d[7] = float(d[7])
            d[8] = ord(d[8][0]) - ord('A')
            d[9] = int(d[9][1])

            if d[11] == "n/a" or d[11][0] == '<':
                temp = 0
            elif d[11][1] == '0':
                temp = 10
            else:
                temp = int(d[11][0])
            d[11] = temp
            d[12] = home_ownership_dict[d[12]]
            d[13] = float(d[13])
            d[14] = income_status_dict[d[14]]
            d[15] = datetime.strptime(d[15], "%b-%Y")
            d[16] = loan_status_dict[d[16]]
            d[17] = ["n", "y"].index(d[17])
            d[23] = state_dict[d[23]]
            d[24] = float(d[24])
            d[25] = int(d[25])
            d[26] = datetime.strptime(d[26], "%b-%Y")
            d[27] = int(d[27])
            # vars 28 and 29 seem to not have incomplete data
            d[30] = int(d[30])
            d[31] = int(d[31])
            d[32] = int(d[32])
            # 33 has incomplete data
            d[34] = int(d[34])
            d[36:42] = map(float, map(lambda x: x.replace(",", ""), d[36:42]))
            d[42:45] = map(float, d[42:45])
            # 45 has incomplete data
            d[46] = float(d[46].replace(",", ""))
            # 47 and 48 have incomplete data
            d[49] = int(d[49])
            d[51] = int(d[51])

            remove_accept_columns(d)
            data.append(d)

    return column_labels, data


def load_clean_accept_data(file_name):
    data = []
    with open(file_name) as f:
        column_labels = f.readline().split("||")

        for line in f:
            d = line.split("||")
            d[:4] = map(int, d[:4])
            d[4:6] = map(float, d[4:6])
            d[6:8] = map(int, d[6:8])
            d[9:11] = map(int, d[9:11])
            d[11] = float(d[11])
            d[12] = int(d[12])
            d[13] = datetime.strptime(d[13], "%Y-%m-%d %H:%M:%S")
            d[14:16] = map(int, d[14:16])
            d[18] = int(d[18])
            d[19] = float(d[19])
            d[20:26] = map(int, d[20:26])
            d[27:37] = map(float, d[27:37])
            d[38:40] = map(int, d[38:40])

            data.append(d)

    return column_labels, data


def load_original_reject_data(file_name, state_dict):
    data = []
    with open(file_name) as f:
        column_labels = map(lambda x: x.strip("\" \n"), f.readline().split(","))
        remove_reject_columns(column_labels)

        for i, line in enumerate(f):
            d = map(lambda x: x.strip("\" \n"), line.split("\",\""))
            try:
                d[0] = float(d[0])
                d[3] = int(d[3])
                d[4] = float(d[4][:-1])
                d[6] = state_dict[d[6]]
                if d[7] == "n/a" or d[7][0] == '<':
                    temp = 0
                elif d[7][1] == '0':
                    temp = 10
                else:
                    temp = int(d[7][0])
                d[7] = temp

                d[8] = int(d[8])

                remove_reject_columns(d)
                data.append(d)
            except (ValueError, KeyError):
                pass

    return column_labels, data

def load_cleaned_reject_data(file_name):
    data = []
    with open(file_name) as f:
        column_labels = f.readline().split("||")

        for line in f:
            d = line.split("||")
            d[0] = float(d[0])
            d[1] = datetime.strptime(d[1], "%Y-%m-%d")
            d[3] = int(d[3])
            d[4] = float(d[4])
            d[5:] = map(int, d[5:])

            assert(len(d) == len(line.split("||")))

            data.append(d)

    return column_labels, data


def write_file(file_name, data, headers=None, separator="||"):
    with open(file_name, "w") as f:
        if headers:
            f.writelines(separator.join(headers) + '\n')

        for row in data:
            f.writelines(separator.join(map(str, row)) + '\n')


def extract_column(data, index):
    return np.array([row[index] for row in data])


def add_column(data, column):
    for row, value in zip(data, column):
        row.append(value)


def generate_histograms(data, column_index, column_name, bin_counts=[10, 20, 30, 40, 50],
                        remove_extreme_values=False, remove_count=100, remove_from_lower=True, normalize=False):
    histogram_data = extract_column(data, column_index)

    print "Min (before removal)= %d" % min(histogram_data)
    print "Max (before removal)= %d" % max(histogram_data)

    if remove_extreme_values:
        if remove_from_lower:
            histogram_data = sorted(histogram_data)[remove_count:]
        else:
            histogram_data = sorted(histogram_data)[:-1 * remove_count]

        print "Min (after removal)= %d" % min(histogram_data)
        print "Max (after removal)= %d" % max(histogram_data)

    for bin_count in bin_counts:
        n, bins, patches = py.hist(histogram_data, bin_count, alpha=0.75, normed=normalize)
        py.title(column_name)
        py.xlim([min(histogram_data) - 1, max(histogram_data) + 1])
        py.xticks(bins, rotation='vertical')
        py.show()


def generate_normalized_state_histogram(data, column_index, state_dict, normalization_factor):
    """

    :param data:
    :param column_index:
    :param state_dict:
    :param normalization_factor: Needed because the number of samples may be really small compared to population. Should
    be 1000 for Accept data and about 250 for reject data
    :return:
    """
    num_states = len(state_dict)
    state_populations_dict = {"AL": 4833722, "AK": 735132, "AZ": 6626624, "AR": 2959373, "CA": 38332521, "CO": 5268367,
                              "CT": 3596080, "DE": 925749, "DC": 646449, "FL": 19552860, "GA": 9992167, "HI": 1404054,
                              "ID": 1612136, "IL": 12882135, "IN": 6570902, "IA": 3090416, "KS": 2893957, "KY": 4395295,
                              "LA": 4625470, "MD": 5928814, "MA": 6692824, "MI": 9895622, "MN": 5420380, "MS": 2991207,
                              "MO": 6044171, "MT": 1015165, "NE": 1868516, "NV": 2790136, "NH": 1323459, "NJ": 8899339,
                              "NM": 2085287, "NY": 19651127, "NC": 9848060, "OH": 11570808, "OK": 3850568,
                              "OR": 3930065, "PA": 12773801, "RI": 1051511, "SC": 4774839, "SD": 844877, "TN": 6495978,
                              "TX": 26448193, "UT": 2900872, "VT": 626630, "VA": 8260405, "WA": 6971406, "WV": 1854304,
                              "WI": 5742713, "WY": 582658, "ME": 1328302, "ND": 723393}

    assert (num_states == len(state_populations_dict))

    histogram_data = extract_column(data, column_index)
    a, b = np.histogram(histogram_data, np.arange(num_states + 1) - 0.5)

    print a
    print sum(a)

    normalized = []
    for state in state_dict.keys():
        normalized_value = a[state_dict[state]] * normalization_factor / float(state_populations_dict[state])
        normalized.append(normalized_value)

    py.bar(range(num_states), normalized, align='center')
    py.xticks(range(num_states))
    py.xlim([-1, num_states + 1])
    py.ylim([0, 1])

    py.show()


def main():
    state_dict = {"AL": 0, "AK": 1, "AZ": 2, "AR": 3, "CA": 4, "CO": 5, "CT": 6, "DE": 7, "DC": 8, "FL": 9, "GA": 10,
                  "HI": 11, "ID": 12, "IL": 13, "IN": 14, "IA": 15, "KS": 16, "KY": 17, "LA": 18, "MD": 19, "MA": 20,
                  "MI": 21, "MN": 22, "MS": 23, "MO": 24, "MT": 25, "NE": 26, "NV": 27, "NH": 28, "NJ": 29, "NM": 30,
                  "NY": 31, "NC": 32, "OH": 33, "OK": 34, "OR": 35, "PA": 36, "RI": 37, "SC": 38, "SD": 39, "TN": 40,
                  "TX": 41, "UT": 42, "VT": 43, "VA": 44, "WA": 45, "WV": 46, "WI": 47, "WY": 48, "ME": 49, "ND": 50}


    # accept data
    # -----------------
    col, data = load_clean_accept_data("cleanedAcceptData.csv")

    # generate_histograms(data, 0, col[0])
    # generate_histograms(data, 11, col[11], remove_extreme_values=True, remove_count=1000, remove_from_lower=False)
    # generate_histograms(data, 22, col[22])
    # generate_histograms(data, 9, col[9], bin_counts=[np.arange(12) - 0.5], normalize=True)
    # generate_normalized_state_histogram(data, 18, state_dict, 1000)
    generate_histograms(data, 4, col[4])

    # reject data
    # -----------------
    # col, data = load_cleaned_reject_data("cleanedRejectData.csv")

    # generate_histograms(data, 6, col[6], bin_counts=[np.arange(12) - 0.5], normalize=True)
    # generate_normalized_state_histogram(data, 5, state_dict, 200)


if __name__ == "__main__":
    main()