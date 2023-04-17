import torch.nn as nn
class Model(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_out):
        super(Model, self).__init__()
        self.n_inputs=n_inputs
        self.n_hidden=n_hidden
        self.n_out=n_out

        self.h1 = nn.Linear(self.n_inputs, n_hidden)
        self.a1 = nn.ReLU()
        self.b1 = nn.BatchNorm1d(n_hidden)

        self.h2 = nn.Linear(n_hidden, n_hidden)
        self.a2 = nn.ReLU()
        self.d2 = nn.Dropout(p=0.1)

        self.h3 = nn.Linear(n_hidden, n_out)
        self.a3 = nn.Sigmoid()

    def forward(self, inp):
        out = self.a1(self.h1(inp))
        out = self.b1(out)
        out = self.a2(self.h2(out))
        out = self.d2(out)
        out = self.a3(self.h3(out))
        return out
