import math
from typing import Any
from tqdm import tqdm
from csv import reader
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, save, tensor
from torch.autograd import Variable
import argparse
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import random_split, SubsetRandomSampler, DataLoader
import time
import csv
from torch import nn
from TCN.tcn import TemporalConvNet
import torch.nn.functional as F

class TCN (nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:,:,-1])#.transpose(1,2)

parser = argparse.ArgumentParser(description='Sequence Modeling - cest')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: true)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.05)0.2 之前模型用')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: 0.2)')
parser.add_argument('--epoch', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=7,
                    help='# of levels (default: 4)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=0.0100,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: RMSprop)')
parser.add_argument('--nhid', type=int, default=150,
                    help='number of hidden units per layer (default: 150)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()
torch.manual_seed(args.seed)

#载入数据并划分数据集
tmp1 = np.loadtxt("./data//CEST/input_x_7_pool_315w_101_2.csv", dtype=np.float32, delimiter=",")
input_datasets = tmp1[0:,0:]#加载数据部分
tmp2 = np.loadtxt("./data//CEST/output_y_7_pool_315w_101_2.csv", dtype=np.float32, delimiter=",")
output_datasets = tmp2[0:,0:]#加载数据部分
input_datasets = input_datasets 
output_datasets = output_datasets
x_shape = input_datasets.shape
y_shape = output_datasets.shape
n = x_shape[0]

train_split = 0.7  
n_train = int(n*train_split)
n_val = n - n_train
x_train = input_datasets[:n_train,0:]
x_val = input_datasets[n_train:,0:]
x_train_length = x_train.shape[0]
x_val_length = x_val.shape[0]

y_train = output_datasets[:n_train,0:]
y_val = output_datasets[n_train:,0:]
y_train_length = y_train.shape[0]
y_val_length = y_val.shape[0]

torch.set_printoptions(precision=5)
x_train = torch.tensor(x_train)
x_val = torch.tensor(x_val)
y_train = torch.tensor(y_train)
y_val = torch.tensor(y_val)

x_train, y_train = Variable(x_train), Variable(y_train)
x_val, y_val = Variable(x_val), Variable(y_val)

x = np.loadtxt("./data//CEST/input_x_invivo_20210515_5.csv", dtype=np.float32, delimiter=",")
test_input_datasets = x[0:,0:]
y = np.loadtxt("./data//CEST/output_y_invivo_20210515_5.csv", dtype=np.float32, delimiter=",")
test_output_datasets = y[0:,0:]

torch.set_printoptions(precision=5) 
x_test = torch.tensor(test_input_datasets)
y_test = torch.tensor(test_output_datasets)

x_test, y_test = Variable(x_test), Variable(y_test)

batch_size = 256
seq_len = 101  
iters = 100
n_classes = 101 
n_train = int(n*train_split)
n_val = n - n_train

input_size = output_size = 101

steps=0
best_acc = 0.0


input_channels = output_channels = 101

channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout

model = TCN(101, 101, channel_sizes, kernel_size, dropout=dropout)

criterion = torch.nn.MSELoss(reduce= True)
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

def evaluate():
    model.train()
    batch_idx = 1
    total_loss = 0
    with torch.no_grad():
        for i in range(0, x_val.size(0), batch_size):
            if i + batch_size > x_val.size(0):
                x = x_val[i:]
                y = y_val[i:]
            else:
                x = x_val[i:(i + batch_size)]
                y = y_val[i:(i + batch_size)]
            output = model(x.unsqueeze(1))
            val_loss = F.mse_loss(output.view(-1, 101), y)  #F.mse_loss   criterion
            return val_loss.item()

def train(epoch):
    model.train()
    batch_idx = 1
    total_loss = 0
    all_loss=0
    for i in range(0, x_train.size(0), batch_size):
        if i + batch_size > x_train.size(0):
            x = x_train[i:]
            y = y_train[i:]
        else:
            x = x_train[i:(i+batch_size)]
            y = y_train[i:(i+batch_size)]
        optimizer.zero_grad()
        output = model(x.unsqueeze(0))#.squeeze(0)
    
        loss = F.mse_loss(output.view(-1, 101), y)  
 
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  

        optimizer.step()
        batch_idx += 1
        total_loss += loss.item()
        all_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            cur_loss = total_loss / args.log_interval
            processed = min(i+batch_size, x_train.size(0))
            print('Train Epoch: {:2d} [{:6d}/{:6d} ({:.0f}%)]\tLearning rate: {:.7f}\tLoss: {:.7f}'.format(
                epoch, processed, x_train.size(0), 100.*processed/x_train.size(0), lr, cur_loss))
            total_loss = 0

def test():
    mylist=[]
    global output
    model.train()
    batch_idx = 1
    total_loss = 0

    with torch.no_grad():
        output = model(x_test.unsqueeze(1)) 
        mylist.append(output.numpy())
        test_loss = F.mse_loss(output.view(-1, 101), y_test)    
        with open('predict-TCN.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerows(output.numpy())
        return test_loss.item()

if __name__ == "__main__":
    best_vloss = 1e-2
    try:
        all_vloss = []
        for epoch in range(1, args.epoch+1):
            epoch_start_time = time.time()
            train(epoch)
            val_loss = evaluate()
            output = model(x_test.unsqueeze(1))
            test_loss = F.mse_loss(output.view(-1, 101), y_test)
            if val_loss < best_vloss:
                with open("model38.pt", 'wb') as f:
                    print('Save model!\n')
                    torch.save(model, f)
                best_vloss = val_loss
            with open('predict-TCN.csv', 'w', newline='') as fp:
                writer = csv.writer(fp)
                writer.writerows(output.numpy())
            if epoch > 5 and val_loss >= max(all_vloss[-3:]):
                lr = lr / 2.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_vloss.append(val_loss)

    except KeyboardInterrupt:

    with open("modelTCN.pt", 'rb') as f:
        model = torch.load(f)

    test_loss = test()
