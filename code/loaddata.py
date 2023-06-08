import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import RandomOverSampler

class TrainDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype('float32')
        self.y = y.astype('float32').values.reshape((len(y), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

class TestDataset(Dataset):
    def __init__(self, X):
        self.X = X.astype('float32')

    def __getitem__(self, idx):
        return self.X[idx]

    def __len__(self):
        return len(self.X)

class LoadData:
    def __init__(self, config):
        self.batch_size = config.batch_size

    def load_data(self,data_path, does_scale=True):

        data = pd.read_pickle(data_path)
        X = data.drop('label', axis=1).drop('index',axis=1)
        y = data['label']
        oversampler = RandomOverSampler(random_state=42)
        X_oversampled, y_oversampled = oversampler.fit_resample(X, y)
        print(y_oversampled.value_counts())
        X_train, X_val, y_train, y_val = train_test_split(X_oversampled, y_oversampled, test_size=0.2, random_state=1)

        if does_scale: 
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
        else:
            scaler = None

        train_ds = TrainDataset(X_train, y_train)
        val_ds = TrainDataset(X_val, y_val)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        return train_dl, val_dl, X_train.shape[1],scaler


    def load_test_data(self,data_path,scaler):
        data = pd.read_pickle(data_path)
        X = data.drop('index',axis=1)
        if scaler:
            X = scaler.transform(X)
        test_ds = TestDataset(X)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size,shuffle=False)
        
        return test_dl,data['index']