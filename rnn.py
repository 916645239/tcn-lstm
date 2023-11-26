import torch
from torch import nn
import numpy as np
import torch
import math
from torch.autograd import Variable
import csv
import matplotlib.pyplot as plt
seq_length = 800

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)
        r_out, hidden = self.rnn(x, hidden)
        r_out = r_out.view(-1, self.hidden_dim)
        output = self.fc(r_out)
        return output, hidden

test_rnn = RNN(input_size=101, output_size=101, hidden_dim=150, n_layers=3)

tmp1 = np.loadtxt("../data//CEST/11_input_x_7_pool_315w_101_2.csv", dtype=np.float32, delimiter=",")
input_datasets = tmp1[0:,0:]
tmp2 = np.loadtxt("../data//CEST/input_x_7_pool_315w_101_2.csv", dtype=np.float32, delimiter=",")

input_datasets=torch.tensor(input_datasets)
output_datasets=torch.tensor(output_datasets)
input_datasets, output_datasets = Variable(input_datasets), Variable(output_datasets)
input_size=101
output_size=101
hidden_dim=150
n_layers=3
rnn = RNN(input_size, output_size, hidden_dim, n_layers)
print(rnn)
criterion = nn.MSELoss()
lr=0.001
optimizer = torch.optim.Adam(rnn.parameters(), 0.0001)


origion_loss=5e-4
text_x = np.loadtxt("../data//CEST/11_input_x_invivo_20210515_5.csv", dtype=np.float32, delimiter=",")
test_input_datasets = text_x[0:,0:]
test_y = np.loadtxt("../data//CEST/input_x_invivo_20210515_5.csv", dtype=np.float32, delimiter=",") #output_y_invivo_20210515_5.csv
test_output_datasets = test_y[0:,0:

test_input_datasets=torch.tensor(test_input_datasets)
test_output_datasets=torch.tensor(test_output_datasets)
test_input_datasets, test_output_datasets = Variable(test_input_datasets), Variable(test_output_datasets)

test_input_datasets = torch.Tensor(test_input_datasets).unsqueeze(0)
test_output_datasets = torch.Tensor(test_output_datasets)

loss_arr=[]
def train(rnn, n_steps, print_every):
    hidden = None
    batch_i=n_steps
    total_loss=0
    all_loss=0
    batch_idx = 1
    origion_loss = 5e-4
    for i in range(0, input_datasets.size(0), n_steps):
        if i + n_steps > input_datasets.size(0):
            x = input_datasets[i:]
            y = output_datasets[i:]
        else:
            x = input_datasets[i:(i+n_steps)]
            y = output_datasets[i:(i+n_steps)]
        x_tensor = torch.Tensor(x).unsqueeze(0) 
        y_tensor = torch.Tensor(y)
        prediction, hidden = rnn(x_tensor, hidden)
        hidden = hidden.data
        loss = criterion(prediction, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_loss += loss.item()
        batch_idx += 1
        if batch_idx % print_every == 0:
            cur_loss = total_loss / print_every
            processed = min(i + n_steps, input_datasets.size(0))
            total_loss = 0
    all_avg_loss=all_loss / batch_idx
    loss_arr.append(all_avg_loss)
    origion_loss = 5e-4
    text_x = np.loadtxt("../data//CEST/11_input_x_invivo_20210515_5.csv", dtype=np.float32, delimiter=",")
    test_input_datasets = text_x[0:, 0:] 
    test_y = np.loadtxt("../data//CEST/input_x_invivo_20210515_5.csv", dtype=np.float32, delimiter=",")
    test_output_datasets = test_y[0:, 0:]  
    test_input_datasets = torch.tensor(test_input_datasets)
    test_output_datasets = torch.tensor(test_output_datasets)
    test_input_datasets, test_output_datasets = Variable(test_input_datasets), Variable(test_output_datasets)
    test_input_datasets = torch.Tensor(test_input_datasets).unsqueeze(0) 
    test_output_datasets = torch.Tensor(test_output_datasets)
    hidden=None
    output_test, hidden = rnn(test_input_datasets, hidden)
    test_loss = criterion(test_output_datasets, output_test)
    if all_avg_loss < origion_loss:
        origion_loss = test_loss
        with open('predictRNN_11_2.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerows(output_test.detach().numpy())
    return rnn
n_steps = 512
print_every = 100
for epoch in range(1, 50):
    trained_rnn = train(rnn, n_steps, print_every)



