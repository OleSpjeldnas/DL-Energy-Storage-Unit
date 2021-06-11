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
testing_data = open('TestingData.txt', 'r')
lin = testing_data.readlines()[1:]
lin = list(map(float, lin))
lines = training_data.readlines()[1:]

training = []
f0 = []
s0 = []
times = list()
prediction_t = torch.tensor(lin, dtype=torch.float32)

for index, data in enumerate(lines):
    k = len(data)
    comma1 = data.find(",")
    comma2 = data.find(",", comma1+1, k)
    times.append([float(data[0:comma1-1])])
    f0.append([float(data[comma1 + 1:comma2])])
    s0.append([float(data[comma2 + 1:k - 1])])

mean_f0 = np.mean(f0)
std_f0 = np.std(f0)
f0 = (f0 - mean_f0)/std_f0

mean_s0 = np.mean(s0)
std_s0 = np.std(s0)
s0 = (s0 - mean_s0)/std_s0


def sliding_windows(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


seq_length = 30
x_train_f0, y_train_f0 = sliding_windows(f0, seq_length)
x_train_s0, y_train_s0 = sliding_windows(s0, seq_length)
#x_test, y_test = sliding_windows(y_test1, seq_length)

x_train_f0 = torch.from_numpy(x_train_f0).float().reshape(179, seq_length)
y_train_f0 = torch.from_numpy(y_train_f0).float()
x_train_s0 = torch.from_numpy(x_train_s0).float().reshape(179, seq_length)
y_train_s0 = torch.from_numpy(y_train_s0).float()


#y_test = torch.from_numpy(y_test).float()


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
        self.fc7 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = torch.relu(self.fc1(input))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        out = self.fc7(x)

        return out


def fit(model, input, y_true, num_epochs):
    train_hist = list()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for _ in range(num_epochs):
        optimizer.zero_grad()
        output = model(input).reshape(-1, )
        loss = criterion(output, y_true.reshape(-1, ))
        loss.backward()
        train_hist.append(loss.item())

        if _ % 500 == 0:
            print("Train Loss:", loss.item())
        optimizer.step()

    return train_hist


hidden_dim = 15
model_f0 = NN(input_dim=x_train_f0.shape[1], output_dim=1, hidden_dim=hidden_dim)
model_s0 = NN(input_dim=x_train_f0.shape[1], output_dim=1, hidden_dim=hidden_dim)

n_epochs = 3000
history_s0 = fit(model=model_s0, num_epochs=n_epochs, input=x_train_s0, y_true=y_train_s0)
history_f0 = fit(model=model_f0, num_epochs=n_epochs, input=x_train_f0, y_true=y_train_f0)

val_array = list(x_train_f0[-1, -seq_length:])
predictions = []
for i in range(1, len(lin)+1):
    x = torch.tensor(val_array, dtype=torch.float32).to("cpu").reshape(1, seq_length)
    x = x.view(1, -1, 1)
    y_pred = model_f0(x.reshape(1, seq_length))
    predictions.append(y_pred.detach().numpy())
    val_array.append(y_pred.detach().numpy())
    val_array = val_array[1:]

predictions_f0 = [item for sublist in predictions for item in sublist]

val_array = list(x_train_s0[-1, -seq_length:])
predictions = []
for i in range(1, len(lin)+1):
    x = torch.tensor(val_array, dtype=torch.float32).to("cpu").reshape(1, seq_length)
    x = x.view(1, -1, 1)
    y_pred = model_s0(x.reshape(1, seq_length))
    predictions.append(y_pred.detach().numpy())
    val_array.append(y_pred.detach().numpy())
    val_array = val_array[1:]
predictions_s0 = [item for sublist in predictions for item in sublist]

train_f0 = model_f0(x_train_f0).data.numpy()
train_s0 = model_s0(x_train_s0).data.numpy()

train_s0 = train_s0*std_s0+mean_s0
train_f0 = train_f0*std_f0+mean_f0

total_s0 = np.concatenate((train_s0, np.array(predictions_s0)*std_s0+mean_s0))
total_f0 = np.concatenate((train_f0, np.array(predictions_f0)*std_f0+mean_f0))


plt.axvline(x=len(lines)-seq_length, c='r', linestyle='--')
plt.plot(total_f0, color='pink')
plt.plot(f0[seq_length:]*std_f0+mean_f0, color='black')
plt.suptitle('f0 Prediction')
plt.show()

plt.axvline(x=len(lines)-seq_length, c='r', linestyle='--')
plt.plot(total_s0, color='pink')
plt.plot(s0[seq_length:]*std_s0+mean_s0, color='black')
plt.suptitle('s0 Prediction')
plt.show()


file = open("f0.txt", "w")
file.truncate(0)
file.close()
np.savetxt("f0.txt", predictions_f0, fmt="%s")

file = open("s0.txt", "w")
file.truncate(0)
file.close()
np.savetxt("s0.txt", predictions_s0, fmt="%s")
#plt.plot(predictions_s0)
#plt.plot(predictions_s0)
#plt.show()