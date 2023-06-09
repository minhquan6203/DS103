import torch.nn as nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_dim, n_hidden, n_out):
        super(CNNModel, self).__init__()
        self.input_dim = input_dim
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=n_hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=n_hidden, out_channels=n_hidden, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(n_hidden, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        out = inp.unsqueeze(2) 
        out = self.conv1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
    


class ResModel(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_out):
        super(ResModel, self).__init__()
        self.input_dim = n_inputs
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.conv1 = nn.Conv1d(in_channels=n_inputs, out_channels=n_hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=n_hidden, out_channels=n_hidden, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(n_hidden, n_out)
        self.fc_skip = nn.Linear(n_inputs, n_out) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        out = inp.unsqueeze(2) 
        out = self.conv1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.view(out.size(0), -1)
        out_skip = inp.view(inp.size(0), -1)  
        out_skip = self.fc_skip(out_skip)  
        out = out + out_skip 
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out