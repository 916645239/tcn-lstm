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

class Gru(nn.Module):

    def __init__(self, input_size, hidden_size=101, output_size=101, num_layers=3):
        super().__init__()

        self.lstm = nn.GRU(input_size, hidden_size, num_layers) 
        self.linear1 = nn.Linear(hidden_size, 101) 
        self.linear2 = nn.Linear(101, output_size) 
        self.gru = nn.GRU(input_size, hidden_size, num_layers)

    def forward(self, _x):
        x, _ = self.gru(_x) 
        s, b, h = x.shape 
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(s, b, -1)
        return x


if __name__ == '__main__':

    # checking if GPU is available
    device = torch.device("cpu")

    if (torch.cuda.is_available()):
        device = torch.device("cuda:0")
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    # 数据读取&类型转换
    data_x = np.array(pd.read_csv('../data//CEST/11_input_x_7_pool_315w_101_2.csv', header=None)).astype('float32')
    data_y = np.array(pd.read_csv('../data//CEST/input_x_7_pool_315w_101_2.csv', header=None)).astype('float32') 

    test_x = np.array(pd.read_csv('../data//CEST/11_input_x_invivo_20210515_5.csv', header=None)).astype('float32')
    test_y = np.array(pd.read_csv('../data//CEST/input_x_invivo_20210515_5.csv', header=None)).astype(
        'float32')  


    # 数据集分割
    data_len = len(data_x)
    t = np.linspace(0, data_len, data_len + 1)

    train_data_ratio = 0.8  # Choose 80% of the data for training
    train_data_len = int(data_len * train_data_ratio)

    train_x = data_x[:train_data_len,0:]
    train_y = data_y[:train_data_len,0:]

    val_x = data_x[train_data_len:,0:]
    val_y = data_y[train_data_len:,0:]

    # ----------------- train -------------------
    INPUT_FEATURES_NUM = 101
    OUTPUT_FEATURES_NUM = 101
    train_x_tensor = train_x  
    train_y_tensor = train_y  

    train_x_tensor = torch.from_numpy(train_x_tensor)
    train_y_tensor = torch.from_numpy(train_y_tensor)

    test_x_tensor = test_x.reshape(-1, 1, INPUT_FEATURES_NUM)


    gru_model = GruRNN(INPUT_FEATURES_NUM, 101, output_size=OUTPUT_FEATURES_NUM, num_layers=3)  
    print('GRU model:', gru_model)
    print('model.parameters:', gru_model.parameters)
    print('train x tensor dimension:', Variable(train_x_tensor).size())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=1e-3)

    prev_loss = 1e-2
    max_epochs = 50

    train_x_tensor = train_x_tensor.to(device)
    val_loss_list = []
    for epoch in range(max_epochs):
        output = gru_model(train_x_tensor) #.to(device)
        loss = criterion(output, train_y_tensor)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < prev_loss:
            torch.save(gru_model.state_dict(), 'gru_model.pt')  
            prev_loss = loss

        if loss.item() < 1e-4:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))

    # ----------------- test -------------------
        gru_model = gru_model.eval()  
        val_x_tensor = val_x.reshape(-1, 1, INPUT_FEATURES_NUM)

        pred_y_for_val = gru_model(val_x_tensor)
        pred_y_for_val = pred_y_for_val.view(-1, OUTPUT_FEATURES_NUM)

        val_loss = criterion(pred_y_for_val, val_y)
        print("val loss：", val_loss.item())
        val_loss_list.append(loss)
        print('val_loss_list', val_loss_list)
        best_loss = 1e-3
        if val_loss < best_loss:
            best_loss = val_loss

            pred_y_for_test = gru_model(test_x_tensor)  
            pred_y_for_test = pred_y_for_test.view(-1, OUTPUT_FEATURES_NUM) 
            test_loss = criterion(pred_y_for_test, test_y)

            with open('predictGRU_11_2.csv', 'w', newline='') as fp:
                writer = csv.writer(fp)
                writer.writerows(pred_y_for_test) #.detach().numpy()
