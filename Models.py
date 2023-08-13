import keras.optimizers
from keras.layers import *
from keras.models import *
from keras import *

# opt = keras.optimizers.Adam(learning_rate=0.005)

def model_CNN():
    cnn_clf = Sequential()
    cnn_clf.add(Conv1D(256, kernel_size=(3),input_shape=(19,1),kernel_initializer='uniform'))
    cnn_clf.add(Flatten())
    cnn_clf.add(Dense(256, activation='relu'))
    cnn_clf.add(Dense(120,activation='tanh'))
    cnn_clf.add(Dense(1,activation='linear'))
    cnn_clf.summary()
    cnn_clf.compile( loss = 'mae',optimizer='Adam',metrics=['accuracy'])
    return cnn_clf

def model_LSTM(j):
    lstm_clf = Sequential()
    lstm_clf.add(Conv1D(j, kernel_size=(3), input_shape=( 19,1),kernel_initializer='uniform',padding='same'))
    lstm_clf.add(LSTM(64,activation='tanh',return_sequences=False,kernel_initializer='random_normal'))
    lstm_clf.add(Dense(1,activation = 'linear'))
    lstm_clf.summary()
    lstm_clf.compile(loss= 'mae', optimizer='Adam',metrics=['accuracy'])
    return lstm_clf