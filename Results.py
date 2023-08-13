import numpy as np
from matplotlib import pyplot
from Models import *



def SDAE(y_test,model_predict):
    n , m = np.shape(y_test)
    sdae = []
    err = y_test[:] - model_predict[:]
    sdae.append(np.sqrt(np.mean(((err - np.mean(err)) ** 2))))
    # for k in range(m):
    #     err = y_test[:,k] - model_predict[:,k]
    #     sdae.append(np.sqrt(np.mean(((err - np.mean(err)) ** 2))))
    return sdae

def Pe(y_true,y_pred):
    # print('*****',np.shape(y_true))
    # y_true = y_true.reshape(-1)
    print('*****', np.shape(y_true))
    # y_pred = y_pred.reshape(-1)
    print(np.shape(y_pred))

    pe = []
    error = abs(y_true - y_pred)
    # print(error)
    # print(np.shape(error))
    n , m = np.shape(error)
    for k in range(m):
        j = 0
        for i in range(n):
            if (error[i,k] >3):
                j = j+1
        pe.append((j/(n))*100)
    # pyplot.figure()
    # pyplot.hist(error)
    # pyplot.show()
    return pe

def Mae(y_true, predictions):
    n , m = np.shape(y_true)
    mae = []
    mae.append(np.mean(np.abs(y_true[:] - predictions[:])))
    # for k in range(m):
    #     mae.append(np.mean(np.abs(y_true[:,k] - predictions[:,k])))
    return mae

def difference(y_true,y_pred):
    diff = abs(y_true-y_pred)
    return diff