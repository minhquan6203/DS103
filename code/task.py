import torch
import torch.nn as nn
import torch.optim as optim
import os

from NN_model import Model,Model2,Skip_Model
from CNN_model import Text_CNNModel
from loaddata import LoadData
from sklearn.metrics import f1_score, confusion_matrix
import pandas as pd
from svm import get_kernel

class Classify_task:
    def __init__(self, config):
        self.num_epochs=config.num_epochs
        self.patience=config.patience
        self.type_model=config.type_model
        self.train_path=config.train_path
        self.test_path=config.test_path
        self.n_hidden=config.n_hidden
        self.batch_size=config.batch_size
        self.learning_rate=config.learning_rate
        self.n_out=config.n_out
        self.save_path=config.save_path
        self.dataloader=LoadData(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train,self.valid,self.n_input,self.scaler= self.dataloader.load_data(data_path=self.train_path)
        self.d_model=config.d_model
        print('n_input: ',self.n_input)
        if self.type_model=='nn':
            self.base_model = Model(n_inputs=self.n_input,d_model=self.d_model,n_hidden=self.n_hidden,n_out=self.n_out).to(self.device)
        if self.type_model=='init':
            self.base_model = Model2(n_inputs=self.n_input,d_model=self.d_model,n_hidden=self.n_hidden,n_out=self.n_out).to(self.device)
        if self.type_model== 'skip':
            self.base_model= Skip_Model(n_inputs=self.n_input,d_model=self.d_model,n_hidden=self.n_hidden,n_out=self.n_out).to(self.device)
        if self.type_model=='cnn':
            self.base_model=Text_CNNModel(n_inputs=self.n_input,d_model=self.d_model,n_hidden=self.n_hidden,n_out=self.n_out).to(self.device)
        if self.type_model=='svm':
            self.gamma = config.gamma
            self.degree = config.degree
            self.kernel_type = config.kernel_type
            self.r = config.r
            self.base_model=get_kernel(self.kernel_type,self.d_model,self.n_out,self.gamma,self.r,self.degree).to(self.device)
        self.linear=nn.Linear(self.n_input,self.d_model)
        self.loss_function =nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate)
    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)
        
        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")

          
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_valid_f1=checkpoint['valid_f1']
        else:
            best_valid_f1 = 0.0
            
        threshold=0
        self.base_model.train()

        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            valid_f1 = 0.0
            train_f1 = 0.0
            train_loss = 0.0
            valid_loss = 0.0
            for inputs, labels in self.train:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.base_model(inputs)
                loss = self.loss_function(output, labels.float())
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_predictions = (output > 0.5).float()
                train_f1 += f1_score(labels.cpu(), train_predictions.cpu(),average='macro')

            with torch.no_grad():
                for inputs, labels in self.valid:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    output = self.base_model(inputs)
                    loss = self.loss_function(output, labels.float())
                    valid_loss += loss.item()
                    valid_predictions = (output > 0.5).float()
                    valid_f1 += f1_score(labels.cpu(), valid_predictions.cpu(),average='macro')

    
            train_loss /= len(self.train)
            train_f1 /= len(self.train)
            valid_loss /= len(self.valid)
            valid_f1 /= len(self.valid)

            print(f"epoch {epoch + 1}:")
            print(f"train Loss: {train_loss:.4f} | train f1-score: {train_f1:.4f}")
            print(f"valid Loss: {valid_loss:.4f} | valid f1-score: {valid_f1:.4f}")


            # save the model state dict
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'valid_f1': valid_f1,
                'train_f1':train_f1,
                'train_loss':train_loss,
                'valid_loss':valid_loss}, os.path.join(self.save_path, 'last_model.pth'))

            # save the best model
            if epoch > 0 and valid_f1 < best_valid_f1:
              threshold+=1
            else:
              threshold=0
            if valid_f1 > best_valid_f1:
                best_valid_f1 = valid_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'valid_f1': valid_f1,
                    'train_f1':train_f1,
                    'train_loss':train_loss,
                    'valid_loss':valid_loss}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with validation f1 score of {valid_f1:.4f}")
            
            # early stopping
            if threshold>=self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break

    def evaluate(self):
        test_data,id = self.dataloader.load_test_data(data_path=self.test_path,scaler=self.scaler)
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'), map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('chưa train model mà đòi test')
        self.base_model.eval()
        pred_labels = []
        with torch.no_grad():
            for inputs in test_data:
                inputs = inputs.to(self.device)
                output = self.base_model(inputs)
                preds=[out[0] for out in (output > 0.5).int().cpu().numpy()]
                pred_labels.extend(preds)
                data = {'index': id, 'label': pred_labels}
        df = pd.DataFrame(data)
        df.to_csv('./submission.csv', index=False)
        print("task done!!!")