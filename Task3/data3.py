import numpy as np
import torch
import torch.nn as nn
import sklearn
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

training_data = open('TrainingData.txt', 'r')
testing_data = open('TestingData.txt', 'r')
lin = testing_data.readlines()[1:]
lin = list(map(float, lin))
lines = training_data.readlines()[1:]

training = []
f0 = []
s0 = []
times = list()
prediction_t = torch.tensor(lin, dtype=float)

for index, data in enumerate(lines):
    k = len(data)
    comma1 = data.find(",")
    comma2 = data.find(",", comma1+1, k)
    times.append([float(data[0:comma1-1])])
    f0.append([float(data[comma1 + 1:comma2])])
    s0.append([float(data[comma2 + 1:k - 1])])
#times = times[8:]
times = (times - np.mean(times))/np.std(times)
f0 = (f0 - np.mean(f0))/np.std(f0)

def sliding_windows(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)
seq_length = 10
x_train1, x_test1, y_train1, y_test1 = sklearn.model_selection.train_test_split(times, f0, train_size=0.8, shuffle=False)
x_train, y_train = sliding_windows(y_train1, seq_length)
x_test, y_test = sliding_windows(y_test1, seq_length)

c = int(0.8*len(f0))
train_input = torch.tensor(f0[:c-1], dtype=torch.float32)
train_output = torch.tensor(f0[1:c], dtype=torch.float32)
test_input = torch.tensor(f0[c:-1], dtype=torch.float32)
test_output = torch.tensor(f0[c+1:], dtype=torch.float32)

print(test_output.shape)
print(test_input.shape)
#print(x_train[0])