import numpy as np
import torch
import torch.nn as nn
from scipy import stats
import sklearn
from sklearn.model_selection import train_test_split
import torch.optim as optim
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

training_101 = open('TrainingData_101.txt', 'r')
training_401 = open('TrainingData_401.txt', 'r')
training_1601 = open('TrainingData_1601.txt', 'r')
testing_data = open('TestingData.txt', 'r')
lin = testing_data.readlines()
lines_101 = training_101.readlines()
lines_401 = training_401.readlines()
lines_1601 = training_1601.readlines()
training_101 = list()
training_401 = list()
training_1601 = list()

for i, word in enumerate(lines_101):
    res_101 = word.split()
    training_101.append(res_101[0:len(res_101)])
    #y_vals_101.append([float(res_101[len(res_101)-1:len(res_101)][0])])
    training_101[len(training_101)-1] = list(map(float, training_101[len(training_101)-1]))


for i, word in enumerate(lines_401):
    res_401 = word.split()
    training_401.append(res_401[0:len(res_401)])
    #y_vals_401.append([float(res_401[len(res_401)-1:len(res_401)][0])])
    training_401[len(training_401)-1] = list(map(float, training_401[len(training_401)-1]))


for i, word in enumerate(lines_1601):
    res_1601 = word.split()
    training_1601.append(res_1601[0:len(res_1601)])
    training_1601[len(training_1601)-1] = list(map(float, training_1601[len(training_1601)-1]))



means = list()
stds = list()
training_101 = np.asmatrix(np.array(training_101))
training_401 = np.asmatrix(np.array(training_401))
training_1601 = np.asmatrix(np.array(training_1601))
test_1601 = []

for i in range(8):
    mean = np.mean(training_101[:, i])
    std = np.std(training_101[:, i])
    training_101[:, i] = (training_101[:, i]-mean)/std
    training_1601[:, i] = (training_1601[:, i]-mean)/std
    training_401[:, i] = (training_401[:, i]-mean)/std
    means.append(mean)
    stds.append(std)

datasets_meshes = []
datasets_meshes.append(training_101)
datasets_meshes.append(training_401)
datasets_meshes.append(training_1601)
training_sets = list()
training_sets.append(datasets_meshes[0])
for l in range(1, 3):
    ns = datasets_meshes[l].shape[0]

    obs_diff = datasets_meshes[l][:ns, -1] - datasets_meshes[l - 1][:ns, -1]
    ts_detail_l = np.concatenate([datasets_meshes[l][:ns, :8], obs_diff.reshape(-1, 1)], 1)
    training_sets.append(ts_detail_l)


class NeuralNet(nn.Module):

    def __init__(self, input_dimension, output_dimension, hidden_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_dimension, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.l10 = nn.Linear(hidden_size, hidden_size)
        self.l11 = nn.Linear(hidden_size, hidden_size)
        self.l12 = nn.Linear(hidden_size, hidden_size)
        self.l13 = nn.Linear(hidden_size, hidden_size)
        self.l21 = nn.Linear(hidden_size, hidden_size)
        self.l22 = nn.Linear(hidden_size, output_dimension)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))
        x = torch.relu(self.l5(x))
        x = torch.relu(self.l10(x))
        x = torch.relu(self.l11(x))
        x = torch.relu(self.l12(x))
        x = torch.relu(self.l13(x))
        x = torch.relu(self.l21(x))

        return self.l22(x)


def init_xavier(model, retrain_seed):
    torch.manual_seed(retrain_seed)
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('tanh')
            #torch.nn.init.xavier_uniform_(m.weight, gain=g)
            torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)
    model.apply(init_weights)


def fit(model, training_set, num_epochs, optimizer, p, verbose=True):
    history = list()

    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose: print("################################ ", epoch, " ################################")

        running_loss = list([0])

        # Loop over batches
        for j, (x_train_, u_train_) in enumerate(training_set):
            def closure():
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                u_pred_ = model(x_train_)
                # Item 1. below
                loss = torch.mean((u_pred_.reshape(-1, ) - u_train_.reshape(-1, )) ** p)
                # Item 2. below
                loss.backward()
                # Compute average training loss over batches for the current epoch
                running_loss[0] += loss.item()
                return loss

            # Item 3. below
            optimizer.step(closure=closure)

            #print('Current Loss: ', (running_loss[0] / len(training_set)))
        history.append(running_loss[0])

    print('Final Loss: ', history[-1])

    return history

def predict(list_models, inputs_):
    output_ = torch.zeros((inputs_.shape[0], 1))
    for i in range(len(list_models)):
        output_ = output_ + list_models[i](inputs_)
    return output_

approximate_models = list()

for i, current_ts in enumerate(training_sets):
    inputs = torch.from_numpy(current_ts[:, :8]).type(torch.float32)
    output = torch.from_numpy(current_ts[:, -1].reshape(-1, 1)).type(torch.float32)
    if i == 2:
        test_1601[:] = current_ts[131:, :]
        inputs = torch.from_numpy(current_ts[:130, :8]).type(torch.float32)
        output = torch.from_numpy(current_ts[:130, -1].reshape(-1, 1)).type(torch.float32)

    batch_size = inputs.shape[0]
    training_set = DataLoader(torch.utils.data.TensorDataset(inputs, output), batch_size=batch_size, shuffle=True)

    model = NeuralNet(input_dimension=inputs.shape[1], output_dimension=output.shape[1], hidden_size=10)
    # Random Seed for weight initialization
    retrain = 128
    # Xavier weight initialization
    init_xavier(model, retrain)

    optimizer_ = optim.LBFGS(model.parameters(), lr=0.5, max_iter=1, max_eval=50000, tolerance_change=1.0 * np.finfo(float).eps)

    n_epochs = 400
    history = fit(model, training_set, n_epochs, optimizer_, p=2, verbose=False)

    #plt.grid(True, which="both", ls=":")
    #plt.plot(np.arange(1, n_epochs + 1), np.log10(history), label="Train Loss")
    #plt.legend()

    approximate_models.append(model)

test_1601 = np.asmatrix(np.array(test_1601))
test_inputs = torch.from_numpy(test_1601[:, :8]).type(torch.float32)
test_output = torch.from_numpy(training_1601[131:, -1]).type(torch.float32).reshape(-1, )


test_pred_ml = predict(approximate_models, test_inputs).reshape(-1, )
err_ml = (torch.mean((test_output - test_pred_ml) ** 2) / torch.mean(test_output ** 2)) ** 0.5

print("Final Error:", err_ml.item())


sobol_points = open('samples_sobol.txt', 'r')
testing_data = open('TestingData.txt', 'r')
lin = sobol_points.readlines()
testing_data_1 = testing_data.readlines()
transformed = list()
sobol = list()
testing_data = list()
for i, word in enumerate(lin):
    res = word.split()
    sobol.append(res[0:len(res)])
    sobol[len(sobol)-1] = list(map(float, sobol[len(sobol)-1]))
for i, word in enumerate(testing_data_1):
    res = word.split()
    testing_data.append(res[0:len(res)])
    testing_data[len(testing_data)-1] = list(map(float, testing_data[len(testing_data)-1]))

transformed = training_101
sobol = np.asmatrix(sobol)
testing_data = np.asmatrix(testing_data)

trafo = np.zeros_like(sobol)
for i in range(0, 8):
    slope, intercept, r_value_0, p_value_0, std_err_0 = stats.linregress(np.asarray(sobol)[:, i], np.asarray(transformed)[:, i])
    testing_data[:, i] = slope * testing_data[:, i] + intercept
    #trafo[:, i] = (trafo[:, i]-means[i])/stds[i]

#print(trafo[0])
trafo_tensor = torch.from_numpy(testing_data).type(torch.float32)
final_result = predict(approximate_models, trafo_tensor).reshape(-1, )

#file = open("2_submission.txt", "w")
#file.truncate(0)
#file.close()
#np.savetxt("2_submission.txt", final_result.detach().numpy(), fmt="%s")



