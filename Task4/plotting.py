
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tsmoothie.smoother import *
import pandas as pd
from scipy.signal import savgol_filter

training_data = open('TrainingData.txt', 'r')
measurements = open('MeasuredData.txt', 'r')
lin = measurements.readlines()
lines = training_data.readlines()

x_training = list()
y_training = list()
measure_data = list()

for _, element in enumerate(lines):
    res = element.split()
    x_training.append([float(res[0]), float(res[2])])
    y_training.append(float(res[1]))
for _, element in enumerate(lin):
    res = element.split()
    measure_data.append([float(res[0]), float(res[1])])


x_training = np.asmatrix(np.array(x_training))
measure_data = np.asmatrix(np.array(measure_data))
batches = list()
for k in range(0, 7):
    batches.append(x_training[128*k+1:128*(k+1)])

for i, batch in enumerate(batches):
    plt.scatter(np.array(batch)[:, 0], np.array(batch)[:, 1])
    plt.suptitle(y_training[128*i+1])
    plt.show()

plt.scatter(np.array(measure_data)[:, 0], np.array(measure_data)[:, 1])
plt.show()
smooth_data = pd.Series(np.array(measure_data)[:, 1]).rolling(window=3).mean()
measure = list()
for i, entry in enumerate(smooth_data):
    if str(entry) != 'nan':
        measure.append([measure_data[i, 0], entry])

plt.plot(smooth_data)
plt.suptitle("Smooth")
plt.show()
print(measure)
print(len(measure))



