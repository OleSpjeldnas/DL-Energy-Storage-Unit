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

device = "cpu"
training_data = open('TrainingData.txt', 'r')
f0_predictions = open('f0.txt', 'r')
s0_predictions = open('s0.txt', 'r')
lin_f = f0_predictions.readlines()[1:]
lin_f = list(map(float, lin_f))
lin_s = s0_predictions.readlines()[1:]
lin_s = list(map(float, lin_s))
lines = training_data.readlines()[1:]


testing_data = open('TestingData.txt', 'r')
lin = testing_data.readlines()[1:]
testing_data = list(map(float, lin))

training = []
f0 = []
s0 = []

for index, data in enumerate(lines):
    k = len(data)
    comma1 = data.find(",")
    comma2 = data.find(",", comma1+1, k)
    f0.append(float(data[comma1 + 1:comma2]))
    s0.append(float(data[comma2 + 1:k - 1]))

f0 = np.array(f0)
s0 = np.array(s0)

total_f0 = np.concatenate((f0, f0_predictions))
total_s0 = np.concatenate((s0, s0_predictions))


final = list()
final.append('t,tf0,ts0')

for i, ting in enumerate(f0_predictions):
    final.append(str(testing_data[i])+","+str(ting)+","+str(s0_predictions[i]))

plt.plot(f0_predictions)
plt.show()