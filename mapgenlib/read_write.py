import os
import numpy as np
import matplotlib.pyplot as plt

"""
Read and write helper
"""

def check_and_create(out_dir):
    if os.path.isdir(out_dir) == False:
        os.mkdir(out_dir)
        
def load_data(trainPath):
    # load image patch arrays
    x_train_sim = np.load(trainPath + "x_train_sim.npy")
    y_train_sim = np.load(trainPath + "y_train_sim.npy")
    x_test_sim = np.load(trainPath + "x_test_sim.npy")
    y_test_sim = np.load(trainPath + "y_test_sim.npy")
    return x_train_sim, y_train_sim, x_test_sim, y_test_sim

def load_batch(x_train_sim, y_train_sim, batch_size):
    total_samples = len(x_train_sim)
    ids = np.arange(total_samples)
    np.random.shuffle(ids)
    n_batches = int(total_samples / batch_size)
    for i in range(n_batches-1):
        batch_idx = ids[i*batch_size:(i+1)*batch_size]
        imgs_A = x_train_sim[batch_idx]
        imgs_B = y_train_sim[batch_idx]
        yield imgs_B, imgs_A

def save_hist(History1, outPath):

    ### Save history
    History1_loss = History1.history['loss']
    History1_acc = History1.history['acc']
    History1_val_loss = History1.history['val_loss']
    History1_val_acc = History1.history['val_acc']

    thefile1 = open(outPath + 'History1_loss.txt', 'w')
    for item in History1_loss:
        thefile1.write("%s\n" % item)
    thefile1.close()

    thefile2 = open(outPath + 'History1_acc.txt', 'w')
    for item in History1_acc:
        thefile2.write("%s\n" % item)
    thefile2.close()

    thefile3 = open(outPath + 'History1_val_loss.txt', 'w')
    for item in History1_val_loss:
        thefile3.write("%s\n" % item)
    thefile3.close()

    thefile4 = open(outPath + 'History1_val_acc.txt', 'w')
    for item in History1_val_acc:
        thefile4.write("%s\n" % item)
    thefile4.close()

    ### Plot history of average covariance - accuracy and loss of the models
    plt.figure()
    plt.plot(History1.history['loss'])
    plt.plot(History1.history['val_loss'])
    plt.title('loss & val_loss')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(outPath + "loss", dpi=1000)

    plt.figure()
    plt.plot(History1.history['acc'])
    plt.plot(History1.history['val_acc'])
    plt.title('acc & val_acc')
    plt.legend(['train', 'test'], loc='lower right')
    plt.savefig(outPath + "acc", dpi=1000)

def save_hist_local(train_acc, train_loss, valid_acc, valid_loss, outPath):    
    
    ### Save history
    History1_loss = train_loss
    History1_acc = train_acc
    History1_val_loss = valid_loss
    History1_val_acc = valid_acc

    thefile1 = open(outPath + 'History1_loss.txt', 'w')
    for item in History1_loss:
        thefile1.write("%s\n" % item)
    thefile1.close()

    thefile2 = open(outPath + 'History1_acc.txt', 'w')
    for item in History1_acc:
        thefile2.write("%s\n" % item)
    thefile2.close()

    thefile3 = open(outPath + 'History1_val_loss.txt', 'w')
    for item in History1_val_loss:
        thefile3.write("%s\n" % item)
    thefile3.close()

    thefile4 = open(outPath + 'History1_val_acc.txt', 'w')
    for item in History1_val_acc:
        thefile4.write("%s\n" % item)
    thefile4.close()