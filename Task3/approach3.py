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
prediction_t = torch.tensor(lin, dtype=torch.float32)
for index, data in enumerate(lines):
    k = len(data)
    comma1 = data.find(",")
    comma2 = data.find(",", comma1+1, k)
    f0.append([float(data[comma1 + 1:comma2])])
    s0.append([float(data[comma2 + 1:k - 1])])

mean_f0 = np.mean(f0)
std_f0 = np.std(f0)
f0 = (f0 - mean_f0)/std_f0

mean_s0 = np.mean(s0)
std_s0 = np.std(s0)
s0 = (s0 - mean_s0)/std_s0

#f0 = torch.tensor(f0, dtype=torch.float32)
#s0 = torch.tensor(s0, dtype=torch.float32)


def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


seq_length = 50
x_f0, y_f0 = sliding_windows(f0, seq_length)
x_s0, y_s0 = sliding_windows(s0, seq_length)
train_size = int(len(y_f0))
#test_size = len(y_f0) - train_size

dataX_f0 = Variable(torch.Tensor(np.array(x_f0)))
dataY_f0 = Variable(torch.Tensor(np.array(y_f0)))

trainX_f0 = Variable(torch.Tensor(np.array(x_f0[0:train_size])))
trainY_f0 = Variable(torch.Tensor(np.array(y_f0[0:train_size])))

#testX_f0 = Variable(torch.Tensor(np.array(x_f0[train_size:len(x_f0)])))
#testY_f0 = Variable(torch.Tensor(np.array(y_f0[train_size:len(y_f0)])))

dataX_s0 = Variable(torch.Tensor(np.array(x_s0)))
dataY_s0 = Variable(torch.Tensor(np.array(y_s0)))

trainX_s0 = Variable(torch.Tensor(np.array(x_s0[0:train_size])))
trainY_s0 = Variable(torch.Tensor(np.array(y_s0[0:train_size])))

#testX_s0 = Variable(torch.Tensor(np.array(x_s0[train_size:len(x_s0)])))
#testY_s0 = Variable(torch.Tensor(np.array(y_s0[train_size:len(y_s0)])))

#valY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=0.5)

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


num_epochs = 1000
learning_rate = 0.001

input_size = 1
hidden_size = 30
num_layers = 1

num_classes = 1

lstm_f0 = LSTM(num_classes, input_size, hidden_size, num_layers)
lstm_s0 = LSTM(num_classes, input_size, hidden_size, num_layers)

# Train the model
def fit(lstm, input, y_true,  num_epochs):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    train_hist = list()
    #test_hist = list()
    for epoch in range(num_epochs):
        if epoch == 8000:
            optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate*0.1)
        outputs = lstm(input)
        #y_pred = lstm(test)
        optimizer.zero_grad()

        loss = criterion(outputs, y_true)
        #test_loss = criterion(y_test, y_pred)
        #test_hist.append(test_loss.item())
        train_hist.append(loss.item())
        loss.backward()

        optimizer.step()
        if epoch % 100 == 0:
            print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    return train_hist

test_hist_f0 = fit(lstm_f0, trainX_f0, trainY_f0, num_epochs)
test_hist_s0 = fit(lstm_s0, trainX_s0, trainY_s0, num_epochs)

train_predict_f0 = lstm_f0(dataX_f0)
data_predict_f0 = train_predict_f0.data.numpy()
dataY_plot_f0 = dataY_f0.data.numpy()


train_predict_s0 = lstm_s0(dataX_s0)

data_predict_s0 = train_predict_s0.data.numpy()
dataY_plot_s0 = dataY_s0.data.numpy()



val_array = list(x_f0[-1, -seq_length:])
predictions = []
for i in range(1, len(lin)+1):
    x = torch.tensor(val_array, dtype=torch.float32).to("cpu")
    x = x.view(1, -1, 1)
    y_pred = lstm_f0(x)
    predictions.append(y_pred.detach().numpy())
    val_array.append(y_pred.detach().numpy())
    val_array = val_array[1:]

predictions_f0 = np.array([item for sublist in predictions for item in sublist])*std_f0+mean_f0
total_f0 = np.concatenate((data_predict_f0*std_f0+mean_f0, predictions_f0))
val_array = list(x_s0[-1, -seq_length:])
predictions = []
for i in range(1, len(lin)+1):
    x = torch.tensor(val_array, dtype=torch.float32).to("cpu")
    x = x.view(1, -1, 1)
    y_pred = lstm_s0(x)
    predictions.append(y_pred.detach().numpy())
    val_array.append(y_pred.detach().numpy())
    val_array = val_array[1:]
predictions_s0 = [item for sublist in predictions for item in sublist]


predictions_s0 = np.array([item for sublist in predictions for item in sublist])*std_s0+mean_s0
total_s0 = np.concatenate((data_predict_s0*std_s0+mean_s0, predictions_s0))
compare = f0[-33:]


plt.axvline(x=len(data_predict_f0), c='r', linestyle='--')
plt.plot(total_f0)
plt.suptitle("f0 Total")
plt.show()

plt.axvline(x=len(data_predict_s0), c='r', linestyle='--')
plt.plot(total_s0)
plt.suptitle("s0 Total")
plt.show()


file = open("f0.txt", "w")
file.truncate(0)
file.close()
#np.savetxt("f0.txt", predictions_f0.detach().numpy(), fmt="%s")

file = open("s0.txt", "w")
file.truncate(0)
file.close()
#np.savetxt("s0.txt", predictions_s0.detach().numpy(), fmt="%s")
