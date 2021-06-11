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

x_training = np.asmatrix(np.array(x_training))
std_t = np.std(np.array(x_training)[:, 0])
mean_t = np.mean(np.array(x_training)[:, 0])

std_u = np.std(np.array(x_training)[:, 1])
mean_u = np.mean(np.array(x_training)[:, 1])

std_T = np.std(np.array(y_training))
mean_T = np.mean(np.array(y_training))
x_training = np.asmatrix(np.array(x_training))
x_training[:, 0] = (x_training[:, 0] - mean_t)/std_t
x_training[:, 1] = (x_training[:, 1] - mean_u)/std_u
v = x_training[:, 1]
t_tensor = torch.tensor(x_training[:, 0], dtype=torch.float32)

y_training = (np.array(y_training)-mean_T)/std_T
measure_data = np.array(measure_data)
#print(measure_data)

train_x = torch.tensor(x_training, dtype=torch.float32)
train_y = torch.tensor(y_training, dtype=torch.float32)
measure_data_tensor = torch.tensor(measure_data, dtype=torch.float32)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(train_x, train_y, train_size=0.8)

def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('tanh')
            #torch.nn.init.xavier_uniform_(m.weight, gain=g)
            torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)
    model.apply(init_weights)


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
num_epochs = 500
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

point = (15.2-mean_u)/std_u
def find_min():
    t_tensor = torch.tensor(x_training[:, 0], dtype=torch.float32)
    t_tensor = t_tensor.reshape(-1, 1)
    final_v = list()
    y_opt = torch.tensor(point, dtype=torch.float32, requires_grad=True).clone().detach()
    y_opt1 = torch.tile(y_opt, t_tensor.shape)
    y_opt1.requires_grad = True
    y_list = list()
    y_list.append(y_opt1)

    optimizer2 = optim.LBFGS([y_opt1], lr=float(0.01), max_iter=50000, max_eval=50000, history_size=100,
                             line_search_fn="strong_wolfe", tolerance_change=1 * np.finfo(float).eps)
    cost = list([0])
    T_tensor = torch.tensor(y_training, dtype=torch.float32).reshape(-1, 1)
    optimizer2.zero_grad()
    # print(t)
    # print(y_opt)
    #input = torch.cat((t, y_opt), dim=1
    def closure():
        optimizer2.zero_grad()
        #y_opt1 = torch.tile(y_opt1, t_tensor.shape).reshape(-1 , 1)
        G = torch.mean((minimizer(torch.cat((t_tensor, y_opt1), dim=1)) - T_tensor))**2
        cost[0] = G
        G.backward()
        return G

    optimizer2.step(closure=closure)
    print(torch.median(y_opt1).detach().numpy()*std_u+mean_u)
    return final_v

final = find_min()
