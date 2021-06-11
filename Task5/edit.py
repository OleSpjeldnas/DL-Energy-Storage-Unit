import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

edit = open('5_baackup2.txt', 'r')
lin_f = edit.readlines()
edit_ = list()
for _, element in enumerate(lin_f):
    res = element.split()
    #edit_.append([float(res[0])*5.066244317445912+11.322010050251258, float(res[1])*99.38047545694239+223.10050251256288])
    edit_.append([float(res[0]), float(res[1])])

edit_ = np.array(edit_)
edit = open('5_backup.txt', 'r')
lin = edit.readlines()
edit1 = list()
for _, element in enumerate(lin):
    res = element.split()
    #edit1.append([float(res[0])*5.066244317445912+11.322010050251258, float(res[1])*99.38047545694239+223.10050251256288])
    edit1.append([float(res[0]), float(res[1])])

edit12 = random.sample(edit1, 1000)
edit12 = np.array(edit12)
plt.scatter(edit_[:, 0], edit_[:, 1])
plt.scatter(edit12[:, 0], edit12[:, 1])
plt.show()
#file = open("5_backup.txt", "w")
#file.truncate(0)
#file.close()
#np.savetxt("5_backup.txt", edit12, fmt="%s")
#x0 = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]).reshape(-1,1)
x0 = list()
for i in range(1500):
    r = np.random.random()*17.8+2.1
    x0.append(r)
x0 = np.array(x0).reshape(-1, 1)
x0 = sorted(x0)
dif_vals = np.array(dif_vals)
#d = dif_vals[:, 0].reshape(-1, 1)
#v = dif_vals[:, 1]
#linreg = LinearRegression().fit(dif_vals[:, 0].reshape(-1, 1), dif_vals[:, 1])

#v_arr = linreg.predict(x0)
#final_linreg = np.zeros((len(v_arr), 2), dtype=float)
#final_linreg[:, 0] = x0
#final_linreg[:, 1] = v_arr
#print(final_linreg)
#plt.plot(x0, v_arr)
#plt.scatter(np.array(edit_)[:, 0], np.array(edit_)[:, 1])
#plt.show()
#print(len(dif_vals))

#plt.scatter(np.array(edit_)[:, 0], np.array(edit_)[:, 1])
#plt.show()

#file = open("5_submission_1.txt", "w")
#file.truncate(0)
#file.close()
#np.savetxt("linreg.txt", final_linreg, fmt="%s")
