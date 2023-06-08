import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearSVM(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        outputs = torch.matmul(x, self.weights.t()) + self.bias
        return outputs


class RBFSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma):
        super(RBFSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        dists = torch.cdist(x, self.weights, p=2)
        dists_normalized = (dists - torch.mean(dists)) / torch.std(dists)
        kernel_matrix = torch.exp(-self.gamma * dists_normalized ** 2)
        outputs = kernel_matrix  + self.bias
        return outputs


class PolySVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, r, degree):
        super(PolySVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.degree = degree
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        # dists = torch.cdist(x, self.weights, p=2)
        # kernel_matrix = (self.gamma * dists + self.r) ** self.degree #này sai công thức nhưng cho kết quả tốt?
        kernel_matrix = (self.gamma * torch.mm(x, self.weights.t()) + self.r) ** self.degree
        outputs = kernel_matrix + self.bias
        return outputs
    

class SigmoidSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma, r):
        super(SigmoidSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        kernel_matrix = torch.tanh(self.gamma * torch.mm(x, self.weights.t())+ self.r)
        outputs = kernel_matrix  + self.bias
        return outputs


class CustomSVM(nn.Module):
    def __init__(self, input_size, num_classes, gamma=0.1, r=1, degree=2):
        super(CustomSVM, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.gamma = gamma
        self.degree = degree
        self.r = r
        self.weights = nn.Parameter(torch.randn(self.num_classes, self.input_size))
        self.bias = nn.Parameter(torch.zeros(self.num_classes))

    def forward(self, x):
        dists = torch.cdist(x, self.weights, p=2)
        kernel_matrix = (self.gamma * dists + self.r) ** self.degree #này sai công thức nhưng cho kết quả tốt?
        outputs = kernel_matrix + self.bias
        return outputs

def get_kernel(kernel_type,input_size ,num_classes, gamma,r, degree):
    if kernel_type == 'linear':
        return LinearSVM(input_size, num_classes)
    elif kernel_type == 'rbf':
        return RBFSVM(input_size, num_classes, gamma)
    elif kernel_type == 'poly':
        return PolySVM(input_size, num_classes, gamma, r, degree)
    elif kernel_type == 'sigmoid':
        return PolySVM(input_size, num_classes, gamma, r)
    elif kernel_type == 'custom':
        return CustomSVM(input_size, num_classes, gamma, r, degree)
    else:
        raise ValueError('không hỗ trợ kernel này')
    