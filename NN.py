import numpy as np
from Models import *
from Results import *



def lstm_model(X,Y,X_t,Y_t,k):
    X = X.reshape([2380, 19,1])
    [m,n]=np.shape(X_t)
    X_t = X_t.reshape([m, 19,1])
    lstm = model_LSTM(k)
    fit_lstm = lstm.fit(X,Y,epochs= 300)#,validation_data=(X_t,Y_t))
    evl_lstm = lstm.evaluate(X_t,Y_t)
    # print(Y_t)
    pre_lstm = lstm.predict(X_t)
    # print(pre_lstm)
    pe_lstm = Pe(Y_t,pre_lstm)
    sdae_lstm = SDAE(Y_t,pre_lstm)
    mae_lstm = Mae(Y_t, pre_lstm)
    diff_lstm = difference(Y_t,pre_lstm)

    return evl_lstm,lstm,fit_lstm,pe_lstm,mae_lstm,sdae_lstm,diff_lstm,pre_lstm


def cnn_model(X,Y,X_t,Y_t):
    X_train_fs = X.reshape([2380, 19,1])
    [m, n] = np.shape(X_t)
    X_test_fs = X_t.reshape([m, 19,1])
    cnn = model_CNN()
    fit_cnn = cnn.fit(X_train_fs, Y,epochs=300)#, validation_data = (X_test_fs,Y_t)
    evl_cnn = cnn.evaluate(X_test_fs, Y_t)
    pre_cnn = cnn.predict(X_test_fs)
    pe_cnn = Pe(Y_t, pre_cnn)
    sdae_cnn = SDAE(Y_t, pre_cnn)
    mae_cnn = Mae(Y_t, pre_cnn)
    diff_cnn = difference(Y_t,pre_cnn)

    return evl_cnn,cnn,fit_cnn,pe_cnn,mae_cnn,sdae_cnn,diff_cnn