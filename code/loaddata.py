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
        self.sampling = config.sampling
        self.scale = config.scale

    def load_data(self,train_path,test_path):

        train = pd.read_pickle(train_path)
        test = pd.read_pickle(test_path)
        #X_train = train.drop(['label', 'index'], axis=1).drop(['GLOBAL_NO','SOUF_RCV_NO','QTUF_RCV_NO','SUBSIDIARY_CD','PRODUCT_ASSORT','HAZARD_FLG','PRODUCT_CD'],axis=1)
        X_train = train.drop(['label', 'index'], axis=1)
        y_train = train['label']
        #X_val = test.drop('index',axis=1).drop(['GLOBAL_NO','SOUF_RCV_NO','QTUF_RCV_NO','SUBSIDIARY_CD','PRODUCT_ASSORT','HAZARD_FLG','PRODUCT_CD'],axis=1)
        X_val = test.drop('index',axis=1)
        y_val =pd.read_csv('/content/sample_submission.csv')['label']
        #X = data.drop('label', axis=1).drop(['index','GLOBAL_NO','SOUF_RCV_NO','QTUF_RCV_NO','SUBSIDIARY_CD','PRODUCT_ASSORT','HAZARD_FLG','PRODUCT_CD'],axis=1)
        if self.sampling:
            oversampler = RandomOverSampler(random_state=42)
            X_train, y_train = oversampler.fit_resample(X_train, y_train)
            X_val, y_val = oversampler.fit_resample(X_val, y_val)
        print(X_train.shape)
        print(y_train.value_counts())
        print(X_val.shape)

        if self.scale: 
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
        else:
            scaler = None

        train_ds = TrainDataset(X_train, y_train)
        val_ds = TrainDataset(X_val, y_val)

        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, shuffle=True)

        return train_dl, val_dl, X_train.shape[1],scaler


    def load_test_data(self,data_path,scaler):
        data = pd.read_pickle(data_path)
        #X = data.drop('index',axis=1).drop(['GLOBAL_NO','SOUF_RCV_NO','QTUF_RCV_NO','SUBSIDIARY_CD','PRODUCT_ASSORT','HAZARD_FLG','PRODUCT_CD'],axis=1)
        X = data.drop('index',axis=1)
        if scaler:
            X = scaler.transform(X)
        test_ds = TestDataset(X)
        test_dl = DataLoader(test_ds, batch_size=self.batch_size,shuffle=False)
        
        return test_dl,data['index']