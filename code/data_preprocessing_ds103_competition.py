import pandas as pd
import numpy as np
import pickle


def drop_col(df,list_col):
    for col in list_col:
        df=df.drop(col,axis=1)
    return df

def map_col(df,list_col):
    for col in list_col:
        mapping={}
        for i in range(len(df[col].unique())):
            mapping[i]=df[col].unique()[i]
        df[col]=df[col].map(mapping)
        df[col]=df[col].astype('category')
    return df

def fill_missing(df):
    df=df.fillna(method='ffill')
    return df

if __name__ == '__main__':
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    list_col_drop=['SUBSIDIARY_CD','OTHER AREA SHIP DIV','WEIGHT_UNIT','SOUF_RCV_NO','QTUF_RCV_NO']
    list_col_map=['Order date','BRAND_CD','PACKING RANK','PRODUCT_CD','DELI_DIV','Ship Mode']
    #xử lý train
    train=fill_missing(train)
    train=drop_col(train,list_col_drop)
    train=map_col(train,list_col_map)
    train.to_pickle('train.pkl')
    #
    test=fill_missing(test)
    test=drop_col(test,list_col_drop)
    test=map_col(test,list_col_map)
    test.to_pickle('test.pkl')
    
    

