import scipy.io as sio
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from torch.autograd import Variable
import math
import csv


class Lstm(nn.Module):
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.linear1 = nn.Linear(hidden_size, output_size) 

    def forward(self, _x):
        x, _ = self.lstm(_x) 
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x


if __name__ == '__main__':

    device = torch.device("cpu")
  
    # 数据读取&类型转换
    data_x = np.array(pd.read_csv('./data/11_input_x_7_pool_315w_101_2.csv', header=None)).astype('float32')
    data_y = np.array(pd.read_csv('./data/input_x_7_pool_315w_101_2.csv', header=None)).astype('float32') 
    test_x = np.array(pd.read_csv('./data/11_input_x_invivo_20210515_5.csv', header=None)).astype('float32')
    test_y = np.array(pd.read_csv('./data/input_x_invivo_20210515_5.csv', header=None)).astype('float32') 
    # 数据集分割
    data_len = len(data_x)
    print(data_len)
    t = np.linspace(0, data_len, data_len)

    train_data_ratio = 0.7 
    train_data_len = int(data_len * train_data_ratio)

    train_x = data_x[0:train_data_len]
    train_y = data_y[0:train_data_len]
    print(train_x.shape)
    print(train_y.shape)
    t_for_training = t[0:train_data_len]

    val_x = data_x[train_data_len:]
    val_y = data_y[train_data_len:]
    print(test_x.shape)
    t_for_testing = t[train_data_len:]
    print(t_for_testing.shape)
    # ----------------- train -------------------
    INPUT_FEATURES_NUM = 101
    OUTPUT_FEATURES_NUM = 101
    train_x_tensor = train_x.reshape(-1, 1, INPUT_FEATURES_NUM)  
    train_y_tensor = train_y.reshape(-1, 1, OUTPUT_FEATURES_NUM) 

    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)
    print(train_x_tensor.shape)

    val_x_tensor = val_x.reshape(-1, 1,
                                 INPUT_FEATURES_NUM)
    val_x_tensor = torch.from_numpy(val_x_tensor) 
    val_x_tensor = val_x_tensor.to(device)

    val_y_tensor = val_y.reshape(-1, 1,
                                 INPUT_FEATURES_NUM)
    val_y_tensor = torch.from_numpy(val_y_tensor)
    val_y_tensor = val_y_tensor.to(device)

    test_x_tensor = test_x.reshape(-1, 1,
                                   INPUT_FEATURES_NUM)
    test_x_tensor = torch.from_numpy(test_x_tensor) 


    lstm_model = Lstm(INPUT_FEATURES_NUM, 150, output_size=OUTPUT_FEATURES_NUM, num_layers=1) 

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-4)

    prev_loss = 1000
    max_epochs = 50

    train_x_tensor = train_x_tensor.to(device)
    batch_size=512
    #epoch=0
    batch_idx = 1
    total_loss = 0
    loss_arr = []
    for epoch in range(max_epochs):
        for i in range(0, train_x_tensor.size(0), batch_size):
            if i + batch_size > train_x_tensor.size(0):
                x = train_x_tensor[i:]
                y = train_y_tensor[i:]
            else:
                x = train_x_tensor[i:(i+batch_size)]
                y = train_y_tensor[i:(i+batch_size)]
            output = lstm_model(x) 
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_idx += 1
            total_loss += loss.item()
            log_interval=100
            lr=0.0001
            if batch_idx % log_interval == 0:
                cur_loss = total_loss / log_interval
                processed = min(i+batch_size, train_x_tensor.size(0))
                print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.7f}\tLoss: {:.7f}'.format(
                epoch, processed, train_x_tensor.size(0), 100.*processed/train_x_tensor.size(0), lr, cur_loss))
                total_loss = 0

        # ----------------- val -------------------
        lstm_model = lstm_model.eval() 

    # prediction on test dataset
        val_x_tensor = val_x.reshape(-1, 1,
                                   INPUT_FEATURES_NUM)
        val_x_tensor = torch.from_numpy(val_x_tensor)
        val_x_tensor = val_x_tensor.to(device)

        val_y_tensor = val_y.reshape(-1, 1,
                                     INPUT_FEATURES_NUM)
        val_y_tensor = torch.from_numpy(val_y_tensor)
        val_y_tensor = val_y_tensor.to(device)
        batch_size = 256
        idx = 0
        total_val_loss = 0
        for i in range(0, val_x_tensor.size(0), 256):
            if i + 256 > val_x_tensor.size(0):
                x = val_x_tensor[i:]
                y = val_y_tensor[i:]
            else:
                x = val_x_tensor[i:(i + 256)]
                y = val_y_tensor[i:(i + 256)]
            val_output = lstm_model(x)
            loss_val = criterion(val_output, y)
            idx += 1
            total_val_loss += loss_val
        val_loss = total_val_loss / idx
        total_val_loss = 0
        idx = 0

        #output_val = lstm_model(val_x_tensor).to(device)
        #print('output_val, val_y_tensor',output_val.shape, val_y_tensor.shape)
        #loss_val = criterion(output_val, val_y_tensor)
        print(" val_Loss:{:.7f}：",  val_loss)


        loss_arr.append(val_loss)
        print('loss_arr:{:.8f}：',loss_arr)

        best_vloss = 1e-2
        if val_loss < best_vloss:
            torch.save(lstm_model.state_dict(), 'lstm_model_11_2.pt')
            best_vloss = val_loss

            pred_y_for_test1 = lstm_model(test_x_tensor) 
            #pred_y_for_test = pred_y_for_test1.view(-1, OUTPUT_FEATURES_NUM) 
            #test_y = torch.from_numpy(test_y)
            pred_y_for_test=pred_y_for_test1.reshape(360,101)
            #print('pred_y_for_test1.shape,pred_y_for_test.shape,test_y.shape,pred_y_for_test', pred_y_for_test1.shape,pred_y_for_test.shape, test_y.shape, pred_y_for_test)
            #loss = criterion(pred_y_for_test, test_y)
            loss = criterion(pred_y_for_test, torch.from_numpy(test_y))
            print("test loss：", loss.item())
            with open('predict_lstm_11_4.csv', 'w', newline='') as fp:
                writer = csv.writer(fp)
                writer.writerows(pred_y_for_test) 

      
