import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import sklearn
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transformed = open('TrainingData_101.txt', 'r')
sobol_points = open('samples_sobol.txt', 'r')
testing_data = open('TestingData.txt', 'r')
lin = sobol_points.readlines()
lines = transformed.readlines()
testing_data = testing_data.readlines()
transformed = list()
sobol = list()

for i, word in enumerate(lines):
    res = word.split()
    transformed.append(res[0:len(res)-1])
    transformed[len(transformed)-1] = list(map(float, transformed[len(transformed)-1]))


for i, word in enumerate(lin):
    res = word.split()
    sobol.append(res[0:len(res)])
    sobol[len(sobol)-1] = list(map(float, sobol[len(sobol)-1]))


transformed = np.asmatrix(np.array(transformed))
sobol = np.asmatrix(sobol)
stds = list()
means = list()

#print(np.asarray(sobol)[:, 0].shape)


for i in range(8):
    mean = np.mean(transformed[:, i])
    std = np.std(transformed[:, i])
    transformed[:, i] = (transformed[:, i]-mean)/std
    means.append(mean)
    stds.append(std)


trafo = np.full_like(transformed, 0)
for i in range(8):
    slope, intercept, r_value_0, p_value_0, std_err_0 = stats.linregress(np.asarray(sobol)[:, i], np.asarray(transformed)[:, i])
    trafo[:, i] = slope * sobol[:, i] + intercept
    trafo[:, i] = trafo[:, i]*stds[i]+means[i]

txtfile=open('2_submission.txt')

L=[]
for line in txtfile:
    L.append(float(line.rstrip()))
txtfile.close()

plt.plot(L[:50])
plt.show()