import keras
import numpy as np
import pandas as pd
import scipy.io as sio
from tcntcn import TCN,tcn_full_summary
from tensorflow.keras.layers import Dense,Input,LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
import time
from keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error

input_train=np.loadtxt("./data//CEST/11_input_x_7_pool_315w_101_2.csv", dtype=np.float32, delimiter=",")
output_train=np.loadtxt("./data//CEST/input_x_7_pool_315w_101_2.csv", dtype=np.float32, delimiter=",")
input_test=np.loadtxt("./data//CEST/11_input_x_invivo_20210515_5.csv", dtype=np.float32, delimiter=",")
output_test=np.loadtxt("./data//CEST/input_x_invivo_20210515_5.csv", dtype=np.float32, delimiter=",")

XTrain1 = input_train
XTrain2 = input_train
XTrain1=np.reshape(XTrain1,(314999,101,1))
XTrain2=np.reshape(XTrain2,(314999,101,1))
YTrain=output_train

XTest1 = input_test
XTest2 = input_test
XTest1=np.reshape(XTest1,(360,101,1))
XTest2=np.reshape(XTest2,(360,101,1))
YTest=output_test

input1 = Input(batch_shape=(None,101,1), name='input1')
input2 = Input(batch_shape=(None,101,1), name='input2')
batch_size, time_steps, input_dim = None, 101, 1
tcn_layer1=TCN(input_shape=(time_steps, input_dim),nb_filters=101,
    kernel_size=3,dilations=[1,2,4,])
x=tcn_layer1(input1)
x=Dense(101)(x)
y=LSTM(64, input_dim=1, input_length=101, return_sequences=True)(input2)
y=LSTM(64, input_dim=1, input_length=101, return_sequences=False)(y)
y=Dense(128,activation='relu')(y)
y=Dense(101)(y)
r=keras.layers.concatenate([x,y])  #,input3  concatenate([x,y])   Average()([x,y])
r = Dense(101, activation='relu')(r)

output = Dense(101,activation='relu',name='main_output')(r)  
model = Model(inputs=[input1,input2], outputs=[output]) 
def get_lr_metric(optimizer): 
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
optimizer = keras.optimizers.Adam(lr=0.001)
lr_metric = get_lr_metric(optimizer)
model.compile(optimizer = optimizer,loss='mse', metrics = ['accuracy',lr_metric])

# Training network
from keras.callbacks import ReduceLROnPlateau
import keras.backend as K
from keras.callbacks import LearningRateScheduler

def scheduler(epoch):
    if epoch % 5 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.5)
        print("lr changed to {}".format(lr * 0.5))
    return K.get_value(model.optimizer.lr)

reduce_lr = LearningRateScheduler(scheduler)
start = time.process_time()
history = model.fit([XTrain1, XTrain2], YTrain, batch_size=512, epochs=50, validation_split=0.2,
                    callbacks=[reduce_lr]) 

# Loss and accuracy
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Trainning acc')
plt.plot(epochs, val_acc, 'b', label='Vaildation acc')
plt.legend()
plt.figure()
plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
plt.legend()
plt.show()

#Evaluate
model.evaluate([XTest1,XTest2],YTest) 

# Predict
YPred = model.predict([XTest1, XTest2]) 

# save data
from scipy.io import savemat
np.savetxt('val_loss',val_loss,fmt='%f',delimiter=',')
np.savetxt('test_Pred',YPred,fmt='%f',delimiter=',')
