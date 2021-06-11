import numpy as np
import torch
import torch.nn as nn
import sklearn
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
testing_data = open('TestingData.txt', 'r')
lin = testing_data.readlines()[1:]
lin = list(map(float, lin))
lines = training_data.readlines()[1:]

f0 = []
s0 = []
prediction_t = torch.tensor(lin, dtype=float)

for index, data in enumerate(lines):
    k = len(data)
    comma1 = data.find(",")
    comma2 = data.find(",", comma1+1, k)
    f0.append([float(data[comma1 + 1:comma2])])
    s0.append([float(data[comma2 + 1:k - 1])])

f0 = torch.tensor(f0, dtype=torch.float32)
f0 = scaler.fit_transform(f0 .reshape(-1, 1))


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


seq_length = 4
x, y = sliding_windows(f0, seq_length)
train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)

        return out


input_size = 1
hidden_size = 2
num_layers = 1

num_classes = 1
PATH = "first_task3.pt"
model = LSTM(num_classes, input_size, hidden_size, num_layers)
model.load_state_dict(torch.load(PATH))
model.eval()

val_array = list(x[-1, -seq_length:])
predictions = []
for i in range(1, len(lin)+1):
    x = torch.tensor(val_array, dtype=torch.float32).to("cpu")
    x = x.view(1, -1, 1)
    y_pred = model(x)
    print("Prediction", i, ":", y_pred.item())
    print("Array", i, ":", val_array)
    predictions.append(y_pred.detach().numpy())
    val_array.append(y_pred.detach().numpy())
    val_array = val_array[1:]

predictions = [item for sublist in predictions for item in sublist]






