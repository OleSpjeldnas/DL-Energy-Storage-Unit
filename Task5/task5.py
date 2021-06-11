
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

training_data = open('TrainingData.txt', 'r')
lines = training_data.readlines()

capacity = list()
d_v = list()

for _, element in enumerate(lines):
    res = element.split()
    d_v.append([float(res[0]), float(res[1])])
    capacity.append(float(res[2]))

mean_d = np.mean(np.array(d_v)[:, 0])
mean_v = np.mean(np.array(d_v)[:, 1])
std_d = np.std(np.array(d_v)[:, 0])
std_v = np.std(np.array(d_v)[:, 1])
#print(mean_d)
#print(mean_v)
#print(std_d)
#print(std_v)
d_v = torch.tensor(d_v, dtype=torch.float32)
d_v[:, 0] = (d_v[:, 0]-torch.mean(d_v[:, 0]))/torch.std(d_v[:, 0])
d_v[:, 1] = (d_v[:, 1]-torch.mean(d_v[:, 1]))/torch.std(d_v[:, 1])
points = d_v.tolist()
capacity = torch.tensor(capacity, dtype=torch.float32)
mean_c = torch.mean(capacity)
std_c = torch.std(capacity)
capacity = (capacity-mean_c)/std_c
proofs = capacity.tolist()
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(d_v, capacity, train_size=0.9)


class MinNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(MinNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        x = self.fc1(input)
        x = torch.relu(x)
        #x = torch.relu(self.fc2(x))
        #x = torch.relu(self.fc3(x))
        #x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        out = self.fc7(x)

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


minimizer = MinNN(input_dim=2, output_dim=1, hidden_dim=20)
retrain = 128
#init_xavier(minimizer, retrain)
num_epochs = 500
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(minimizer.parameters(), lr=0.1)

def train(lstm, input, y_true, num_epochs):
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = lstm(input)
        y_pred = lstm(x_test)

        #print("Test Loss ", epoch, ": ", test_loss.item())
        loss = criterion(outputs.reshape(-1, ), y_true.reshape(-1, ))
        loss.backward()

        optimizer.step()
        #if epoch % 100 == 0:
        #print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    #return train_hist, test_hist

train(minimizer, d_v, capacity, num_epochs)

#print("Hello ", minimizer(torch.tensor([(5.4377471272111-mean_d)/std_d, (130.1062362845787-mean_v)/std_v], dtype=torch.float32))*std_c+mean_c)
edit = open('linreg.txt', 'r')
lin_f = edit.readlines()
edit_ = list()
for _, element in enumerate(lin_f):
    res = element.split()
    edit_.append([(float(res[0])-mean_d)/std_d, (float(res[1])-mean_v)/std_v])

edit_ = np.array(edit_)


final_reg = list()
for _, point in enumerate(edit_):
    y_opt = torch.tensor(point, requires_grad=True, dtype=torch.float32)
    optimizer2 = optim.LBFGS([y_opt], lr=float(0.05), max_iter=50000, max_eval=50000, history_size=100,
                             line_search_fn="strong_wolfe", tolerance_change=1 * np.finfo(float).eps)
    #y_init = torch.clone(y_opt)
    # print(y_init, " init")
    optimizer2.zero_grad()
    cost = list([0])
    flux_ref = 0.45

    def closure():
        G = (minimizer(y_opt) * std_c + mean_c - flux_ref) ** 2
        cost[0] = G
        G.backward()
        return G


    optimizer2.step(closure=closure)

    pred = minimizer(y_opt) * std_c + mean_c
    if abs(pred.item() - 0.45) ** 2 < 1e-7 and 2 <= y_opt[0].item() * std_d + mean_d <= 20 and 50 <= y_opt[
        1].item() * std_v + mean_v <= 400:
        final_reg.append(y_opt.detach().numpy())

file = open("5_backup.txt", "w")
file.truncate(0)
file.close()
np.savetxt("5_backup.txt", final_reg, fmt="%s")