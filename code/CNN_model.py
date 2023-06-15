import torch.nn as nn
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_dim, d_model, n_hidden, n_out):
        super(CNNModel, self).__init__()
        self.linear = nn.Linear(input_dim,d_model)

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=n_hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=n_hidden, out_channels=n_hidden, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(n_hidden, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        out = self.dropout(self.linear(inp))
        out = out.unsqueeze(2) 
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
    

class Text_CNNModel(nn.Module):
    def __init__(self, n_inputs, d_model, n_hidden, n_out):
        super(Text_CNNModel, self).__init__()
        self.linear = nn.Linear(n_inputs, d_model)
        xavier_uniform_(self.linear.weight)
             
        self.conv1 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.relu1 = nn.ReLU()
        kaiming_uniform_(self.conv1.weight, nonlinearity='relu')

        self.conv2 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.relu2 = nn.ReLU()
        kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        
        self.conv3 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(n_hidden)
        self.relu3 = nn.ReLU()
        kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        
        self.conv4 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(n_hidden)
        self.relu4 = nn.ReLU()
        kaiming_uniform_(self.conv4.weight, nonlinearity='relu')
                
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(4 * n_hidden, n_out)
        
        xavier_uniform_(self.fc1.weight)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(2)
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x2 = self.relu2(self.bn2(self.conv2(x)))
        x3 = self.relu3(self.bn3(self.conv3(x)))
        x4 = self.relu4(self.bn4(self.conv4(x)))

        pool1 = F.max_pool1d(x1, x1.shape[2])
        pool2 = F.max_pool1d(x2, x2.shape[2])
        pool3 = F.max_pool1d(x3, x3.shape[2])
        pool4 = F.max_pool1d(x4, x4.shape[2])
        out = torch.cat([pool1, pool2, pool3, pool4], dim=1)
        out = out.squeeze(2) 
        out = self.dropout(self.fc1(out))      
        return out
    


class GAU_CNNModel(nn.Module):
    def __init__(self, n_inputs, d_model, n_hidden, n_out):
        super(GAU_CNNModel, self).__init__()
        self.linear = nn.Linear(n_inputs, d_model)
        self.conv1 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.relu1 = nn.ReLU()
        self.gau1 = nn.GRUCell(n_hidden, n_hidden)  

        self.conv2 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.relu2 = nn.ReLU()
        self.gau2 = nn.GRUCell(n_hidden, n_hidden)  

        self.conv3 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(n_hidden)
        self.relu3 = nn.ReLU()
        self.gau3 = nn.GRUCell(n_hidden, n_hidden)  

        self.conv4 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(n_hidden)
        self.relu4 = nn.ReLU()
        self.gau4 = nn.GRUCell(n_hidden, n_hidden)  

        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(4 * n_hidden, n_out)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(2)
        
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.gau1(x1.squeeze(2), x1.squeeze(2)) 
        x1 = x1.unsqueeze(2)
        
        x2 = self.relu2(self.bn2(self.conv2(x)))
        x2 = self.gau2(x2.squeeze(2), x2.squeeze(2)) 
        x2 = x2.unsqueeze(2)
        
        x3 = self.relu3(self.bn3(self.conv3(x)))
        x3 = self.gau3(x3.squeeze(2), x3.squeeze(2)) 
        x3 = x3.unsqueeze(2)
        
        x4 = self.relu4(self.bn4(self.conv4(x)))
        x4 = self.gau4(x4.squeeze(2), x4.squeeze(2)) 
        x4 = x4.unsqueeze(2)

        pool1 = F.max_pool1d(x1, x1.shape[2])
        pool2 = F.max_pool1d(x2, x2.shape[2])
        pool3 = F.max_pool1d(x3, x3.shape[2])
        pool4 = F.max_pool1d(x4, x4.shape[2])
        
        out = torch.cat([pool1, pool2, pool3, pool4], dim=1)
        out = out.squeeze(2)
        out = self.dropout(self.fc1(out))
        return out


class Skip_CNNModel(nn.Module):
    def __init__(self, n_inputs, d_model, n_hidden, n_out):
        super(Skip_CNNModel, self).__init__()
        self.linear = nn.Linear(n_inputs, d_model)
        xavier_uniform_(self.linear.weight)

        self.conv1 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(n_hidden)
        self.relu1 = nn.ReLU()
        self.gau1 = nn.GRUCell(n_hidden, n_hidden)  
        kaiming_uniform_(self.conv1.weight, nonlinearity='relu')

        self.conv2 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(n_hidden)
        self.relu2 = nn.ReLU()
        self.gau2 = nn.GRUCell(n_hidden, n_hidden)  
        kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        
        self.conv3 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(n_hidden)
        self.relu3 = nn.ReLU()
        self.gau3 = nn.GRUCell(n_hidden, n_hidden)  
        kaiming_uniform_(self.conv3.weight, nonlinearity='relu')
        
        self.conv4 = nn.Conv1d(d_model, n_hidden, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(n_hidden)
        self.relu4 = nn.ReLU()
        self.gau4 = nn.GRUCell(n_hidden, n_hidden)  
        kaiming_uniform_(self.conv4.weight, nonlinearity='relu')
        
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(4 * n_hidden, n_out)
        xavier_uniform_(self.fc1.weight)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(2)
        
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.gau1(x1.squeeze(2), x1.squeeze(2)) 
        x1 = x1.unsqueeze(2)
        
        x2 = self.relu2(self.bn2(self.conv2(x)))
        x2 = self.gau2(x2.squeeze(2), x2.squeeze(2)) 
        x2 = x2.unsqueeze(2)
        
        x3 = self.relu3(self.bn3(self.conv3(x)))
        x3 = self.gau3(x3.squeeze(2), x3.squeeze(2)) 
        x3 = x3.unsqueeze(2)
        
        x4 = self.relu4(self.bn4(self.conv4(x)))
        x4 = self.gau4(x4.squeeze(2), x4.squeeze(2)) 
        x4 = x4.unsqueeze(2)
        
        residual1 = x1
        x1 = x1 + residual1 
        
        residual2 = x2
        x2 = x2 + residual2 
        
        residual3 = x3
        x3 = x3 + residual3 
        
        residual4 = x4
        x4 = x4 + residual4 

        pool1 = F.max_pool1d(x1, x1.shape[2])
        pool2 = F.max_pool1d(x2, x2.shape[2])
        pool3 = F.max_pool1d(x3, x3.shape[2])
        pool4 = F.max_pool1d(x4, x4.shape[2])

        out = torch.cat([pool1, pool2, pool3, pool4], dim=1)
        out = out.squeeze(2)
        out = self.dropout(self.fc1(out))
        return out

