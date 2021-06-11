import numpy as np
import torch
import torch.nn as nn
import sklearn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

scaler = MinMaxScaler(feature_range=(-1, 1))
training_data = open('TrainingData.txt', 'r')
measurements = open('MeasuredData.txt', 'r')
lin = measurements.readlines()
lines = training_data.readlines()

x_training = list()
y_training = list()
measure_data = list()

for _, element in enumerate(lines):
    res = element.split()
    x_training.append([float(res[0]), float(res[1])])
    y_training.append(float(res[2]))
for _, element in enumerate(lin):
    res = element.split()
    measure_data.append([float(res[0]), float(res[1])])


std_t = np.std(np.array(x_training)[:, 0])
mean_t = np.mean(np.array(x_training)[:, 0])

std_u = np.std(np.array(x_training)[:, 1])
mean_u = np.mean(np.array(x_training)[:, 1])

std_T = np.std(np.array(y_training))
mean_T = np.mean(np.array(y_training))
x_training = np.asmatrix(np.array(x_training))
x_training[:, 0] = (x_training[:, 0] - mean_t)/std_t
x_training[:, 1] = (x_training[:, 1] - mean_u)/std_u

y_training = (np.array(y_training)-mean_T)/std_T
measure_data = np.array(measure_data)
#print(measure_data)

train_x = torch.tensor(x_training, dtype=torch.float32)
train_y = torch.tensor(y_training, dtype=torch.float32)
measure_data_tensor = torch.tensor(measure_data, dtype=torch.float32)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(train_x, train_y, train_size=0.8)

class MinNN(nn.Module):
    def __init__(self, hidden_dim):
        super(MinNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, 1)

    def forward(self, input):
        x = self.fc1(input)
        x = torch.relu(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        out = self.fc7(x)

        return out

minimizer = MinNN(hidden_dim=30)
#init_xavier(minimizer, 128)
num_epochs = 200
optimizer = optim.Adam(minimizer.parameters(), lr=0.05)
criterion = nn.MSELoss()
train_hist = list()
test_hist = list()


for _ in range(num_epochs):
    optimizer.zero_grad()
    y_pred = minimizer(x_train).reshape(-1, )
    loss = criterion(y_pred, y_train)
    y_pred_test = minimizer(x_test).reshape(-1, )
    test_loss = criterion(y_pred_test, y_test)
    train_hist.append(loss.item())
    test_hist.append(test_loss.item())

    if _ % 100 == 0:
        print("Train Loss:", loss.item())
        print("Test Loss:", test_loss.item())
    loss.backward()
    optimizer.step()

smooth_data = pd.Series(np.array(measure_data)[:, 1]).rolling(window=3).mean()
measure = list()
for i, entry in enumerate(smooth_data):
    if str(entry) != 'nan':
        measure.append([measure_data[i, 0], entry])

measure_data = np.asmatrix(np.array(measure_data))
measure_data[:, 0] = (measure_data[:, 0]-mean_t)/std_t
measure_data[:, 1] = (measure_data[:, 1]-mean_T)/std_T
test_values = list()
val = 15.1
test_t1 = np.array(measure_data)[:, 0]
test_T1 = torch.tensor(measure_data[:, 1], dtype=torch.float32)
v_arr1 = list()
l_arr1 = list()

for i in range(3000):
    t_val1 = ((val-mean_u)/std_u) *np.ones(test_t1.shape[0]).reshape(-1, )
    s1 = torch.tensor(np.column_stack((test_t1.reshape(-1, ), t_val1)), dtype=torch.float32)
    y_pred1 = minimizer(s1).view((test_t1.shape[0], 1))
    loss1 = criterion(y_pred1, test_T1).item()
    v_arr1.append(val)
    l_arr1.append(loss1)
    val += 0.0001
    #test_values.append([val, loss])

min_i = np.argmin(np.array(l_arr1))
min_v = v_arr1[min_i]
print(min_v)

