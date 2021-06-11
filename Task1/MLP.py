import numpy as np
import torch
import torch.nn as nn
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

scaler = MinMaxScaler(feature_range=(-1, 1))
training_data = open('TrainingData1.txt', 'r')
testing_data = open('TestingData.txt', 'r')
lin = testing_data.readlines()[1:]
lin = list(map(float, lin))
lines = training_data.readlines()[1:]
times = []
f0 = []
s0 = []
validation = []
for index, data in enumerate(lin):
    validation.append([float(data)])
for index, data in enumerate(lines):
    k = len(data)
    comma1 = data.find(",")
    comma2 = data.find(",", comma1+1, k)
    times.append([float(data[0:comma1])])
    f0.append([float(data[comma1 + 1:comma2])])
    s0.append([float(data[comma2 + 1:k - 1])])


times = torch.tensor(times, dtype=torch.float32)
validation = torch.tensor(validation, dtype=torch.float32)

mean_f0 = np.mean(f0)
std_f0 = np.std(f0)
f0 = (f0 - mean_f0)/std_f0

mean_s0 = np.mean(s0)
std_s0 = np.std(s0)
s0 = (s0 - mean_s0)/std_s0

f0 = torch.tensor(f0, dtype=torch.float32)
s0 = torch.tensor(s0, dtype=torch.float32)

validation = (validation-torch.mean(times))/torch.std(times)
times = (times-torch.mean(times))/torch.std(times)


x_train_f0, x_test_f0, y_train_f0, y_test_f0 = sklearn.model_selection.train_test_split(times, f0, train_size=0.9)
x_train_s0, x_test_s0, y_train_s0, y_test_s0 = sklearn.model_selection.train_test_split(times, s0, train_size=0.9)


class NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(NN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = self.fc1(input)
        x = torch.relu(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        x = torch.relu(self.fc7(x))
        out = self.fc8(x)

        return out


def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('tanh')
            #torch.nn.init.xavier_uniform_(m.weight, gain=g)
            torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)
    model.apply(init_weights)

def fit(model, input, y_true, test, y_test, num_epochs):
    train_hist = list()
    test_hist = list()
    optimizer = optim.Adam(model.parameters(), lr=0.004)
    criterion = nn.MSELoss()

    for _ in range(num_epochs):
        if _ == 4000:
            optimizer = optim.Adam(model.parameters(), lr=0.001)

        optimizer.zero_grad()
        output = model(input)
        y_pred = model(test)
        loss = criterion(output, y_true)
        #test_loss = criterion(y_pred, y_test)
        test_loss = (torch.mean((y_pred - y_test) ** 2) / torch.mean(y_pred ** 2)) ** 0.5
        loss.backward()
        train_hist.append(loss.item())
        test_hist.append(test_loss.item())

        if _ % 100 == 0:
            print("Train Loss:", _, " ", loss.item())
            print("Test Loss:", test_loss.item())
        optimizer.step()

    return train_hist, test_hist


hidden_dim = 40
num_epochs = 5000
model_f0 = NN(1, 1, hidden_dim)
retrain = 128
# Xavier weight initialization
init_xavier(model_f0, retrain)

train_hist_f0, test_hist_f0 = fit(model_f0, x_train_f0, y_train_f0, x_test_f0, y_test_f0, num_epochs)

final_f0 = model_f0(times).reshape(-1, ).detach()
plt.plot(times, final_f0, 'r')
plt.plot(times, f0, 'b')
plt.suptitle("Expected vs Correct f0")
plt.show()

y_pred_f0 = model_f0(validation).reshape(-1, ).detach()*std_f0+mean_f0
plt.plot(validation, y_pred_f0, 'r')
plt.suptitle("Network Predictions")#plt.show()

file = open("f0.txt", "w")
file.truncate(0)
file.close()
np.savetxt("f0.txt", y_pred_f0.detach().numpy(), fmt="%s")

