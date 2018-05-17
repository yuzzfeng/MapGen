import matplotlib.pyplot as plt
import numpy as np

acc_train = np.loadtxt("History1_acc.txt")
acc_test = np.loadtxt("History1_val_acc.txt")

loss_train = np.loadtxt("History1_loss.txt")
loss_test = np.loadtxt("History1_val_loss.txt")

plt.figure()
plt.plot(range(len(acc_train)), acc_train, 'r')
plt.plot(range(len(acc_train)), acc_test, 'g')


plt.figure()
plt.plot(range(len(acc_train)), loss_train, 'r')
plt.plot(range(len(acc_train)), loss_test, 'g')
plt.show()