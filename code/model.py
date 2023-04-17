import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.svm import SVC
from sklearn.cluster import KMeans

class CNN_Model(nn.Module): #this repo use ResNet34

    def __init__(self, config):
        super(CNN_Model, self).__init__()
        self.num_classes = config.num_classes
        self.resnet = models.resnet34(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.softmax(x, dim=1)
        return x

class SVM_Model(nn.Module):
    def __init__(self, config):
        super(SVM_Model, self).__init__()
        self.linear = nn.Linear(config.image_H * config.image_W * config.image_C, config.num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)
