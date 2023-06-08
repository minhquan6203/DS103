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

def to_num(df,col_list):
    for col in col_list:
        df[col]=pd.to_numeric(df[col], errors='coerce')
        return df
    
def to_datetime(df,col_list):
    for col in col_list:
        df[col]=pd.to_datetime(df[col])
    df['count_day']=(df['VSD']-df['Order date']).dt.days
    return df

if __name__ == '__main__':
    train = pd.read_csv('./train.csv')
    train = train.replace(r'^\s*$',0, regex=True)
    test = pd.read_csv('./test.csv')
    test = test.replace(r'^\s*$',0, regex=True)

    list_col_to_datetime=['Order date','VSD']
    list_col_to_num=['Consider count hodiday Saturday']
    list_col_drop=['SUBSIDIARY_CD','WEIGHT_UNIT','HAZARD_FLG','PACK QTY','REASON_CD','SOUF_RCV_NO','QTUF_RCV_NO','PRODUCT_ASSORT']
    list_col_map=['BRAND_CD','INNER_CD','PACKING RANK','PRODUCT_CD','DELI_DIV','Ship Mode','OTHER AREA SHIP DIV']
    
    #xử lý train
    train=fill_missing(train)
    train=to_datetime(train,list_col_to_datetime)
    train=drop_col(train,list_col_drop)
    train=to_num(train,list_col_to_num)
    train=map_col(train,list_col_map)
    train.to_pickle('train.pkl')
    #xử lý test
    test=fill_missing(test)
    test=to_datetime(test,list_col_to_datetime)
    test=drop_col(test,list_col_drop)
    test=to_num(test,list_col_to_num)
    test=map_col(test,list_col_map)
    test.to_pickle('test.pkl')
    