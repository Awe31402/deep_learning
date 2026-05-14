import csv
from numpy import asarray

def load_data_1d(filename):
    # load the dataset as a numpy array
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # skip the header
        data = []
        for row in csv_reader:
            data.append(row)
        data = asarray(data).astype(float)
    x = data[:, 0]  # first column is the feature
    y = data[:, 1]  # second column is the label
    return (x, y)

def load_data_2d(filename):
    # load the dataset as a numpy array
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # skip the header
        data = []
        for row in csv_reader:
            data.append(row)
        data = asarray(data).astype(float)
    x = data[:, 0:2]  # first two columns are the features
    y = data[:, 2]   # third column is the label
    return (x, y)