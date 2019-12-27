import random

import mnist_loader
import numpy as np
import network2
import csv
import math
import pandas as pd

def get_row_target(num):
    if num == 0:
        return [1, 0]
    return [0, 1]

training_data = []
validation_data = []
test_data = []
with open('bla_5.csv', 'rb') as csvfile:
    data_new = pd.read_csv(csvfile)
    data_new = data_new.dropna(axis=0,how="any")
    data_list = data_new.values.tolist()

    # data_list = list(data_raw)[1:]
    data_list_new = [row[3:18] for row in data_list]
    num_cols = len(data_list_new[0])
    # data_list_no_null = [row for row in data_list if len([col for col in row if len(col) == 0]) == 0]

    # data_floats = [[float(row[i]) for i in range(0, num_cols)] for row in data_list_new]
    cols = [row for row in list(zip(*data_list_new))]
    means = [np.mean(row) for row in cols]
    variances = [np.var(row) for row in cols]
    new_cols = [(row - np.mean(row)) / math.sqrt((np.var(row))) for row in cols]
    data = list(zip(*new_cols))

    for row_num, row in enumerate(data):
        if row_num == 0:
            continue
        if row_num > 40000:
            break
        npa = np.asarray(row, dtype=np.float32).reshape(-1, 1)
        if row_num <= 20000:
            row_target = get_row_target(data_list[row_num][19])
            npa_target = np.asarray(row_target, dtype=np.float32).reshape(-1, 1)
            training_data.append((npa, npa_target))
        elif row_num <= 10000:
            row_target = data_list[row_num][19]
            npa_target = np.asarray(row_target, dtype=np.float32).reshape(-1, 1)
            validation_data.append((npa, npa_target))
        else:
            row_target = data_list[row_num][19]
            npa_target = np.asarray(row_target, dtype=np.float32).reshape(-1, 1)
            test_data.append((npa, npa_target))


def mean(*args):
    return sum(args) / len(args)


def zero_or_one():
    rand = random.random()
    if rand >= 0.5:
        return 1
    return 0


def target(list):
    if sum(list) <= 7:
        return 0
    return 1

def target_list(list):
    if sum(list) <= 7:
        return [1, 0]
    return [0, 1]




# training_lists = [[zero_or_one() for i in range(0, 15)] for j in range(0, 20000)]
# training_tuples = [(list, target_list(list)) for list in training_lists]
# training_data = [(np.asarray(x).reshape(-1, 1), np.asarray(y).reshape(-1, 1)) for x, y in training_tuples]
#
# validation_lists = [[zero_or_one() for i in range(0, 15)] for j in range(0, 10000)]
# validation_tuples = [(list, target(list)) for list in validation_lists]
# validation_data = [(np.asarray(x).reshape(-1, 1), np.asarray(y).reshape(-1, 1)) for x, y in validation_tuples]
#
# test_lists = [[zero_or_one() for i in range(0, 15)] for j in range(0, 10000)]
# test_tuples = [(list, target(list)) for list in test_lists]
# test_data = [(np.asarray(x).reshape(-1, 1), np.asarray(y).reshape(-1, 1)) for x, y in test_tuples]

# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = network2.Network([15, 10, 5, 2], cost=network2.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 15000, 1000, 0.03, lmbda=1, evaluation_data=test_data, monitor_evaluation_accuracy=True)


