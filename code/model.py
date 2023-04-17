import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.init import kaiming_uniform_, xavier_uniform_

class CNN_Model(nn.Module):
    def __init__(self, n_inputs, n_hidden, num_classes):
        super(CNN_Model, self).__init__()
        self.n_inputs=n_inputs
        self.n_hidden=n_hidden
        self.num_classes=num_classes

        self.h1 = nn.Linear(self.n_inputs, n_hidden)
        kaiming_uniform_(self.h1.weight, nonlinearity='relu')
        self.a1 = nn.ReLU()
        self.b1 = nn.BatchNorm1d(n_hidden)

        self.h2 = nn.Linear(n_hidden, n_hidden)
        kaiming_uniform_(self.h2.weight, nonlinearity='relu')
        self.a2 = nn.ReLU()
        self.d2 = nn.Dropout(p=0.1)

        self.h3 = nn.Linear(n_hidden, num_classes)
        xavier_uniform_(self.h3.weight)
        self.a3 = nn.Sigmoid()

    def forward(self, inp):
        out = self.a1(self.h1(inp))
        out = self.b1(out)
        out = self.a2(self.h2(out))
        out = self.d2(out)
        out = self.a3(self.h3(out))
        return out