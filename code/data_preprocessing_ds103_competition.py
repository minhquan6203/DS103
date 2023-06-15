import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sqlalchemy.sql.expression import asc

def drop_col(df,list_col):
    for col in list_col:
        df=df.drop(col,axis=1)
    return df

def label_encoder(df):
    label_encoder = LabelEncoder()
    for column in df.columns:
        if column != 'index' and (df[column].dtype == 'object' or df[column].dtype == 'datetime64[ns]'):
            mixed_values = False
            for value in df[column]:
                if isinstance(value, str):
                    mixed_values = True
                    break

            if mixed_values:
                df[column] = df[column].astype(str)
            df[column] = label_encoder.fit_transform(df[column])
    return df, label_encoder


def label_encoder_test(df,label_encoder):
    for column in df.columns:
        if column != 'index' and (df[column].dtype == 'object' or df[column].dtype == 'datetime64[ns]'):
            mixed_values = False
            for value in df[column]:
                if isinstance(value, str):
                    mixed_values = True
                    break

            if mixed_values:
                df[column] = df[column].astype(str)
            df[column] = label_encoder.fit_transform(df[column])
    return df


def map_col_test(df,list_col,map_list):
    for i in range(len(list_col)):
        df[list_col[i]]=df[list_col[i]].map(map_list[i])
        df[list_col[i]]=df[list_col[i]].astype('category')
    return df

def fill_missing(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col]=df[col].fillna(mode_value)
    return df
    
def astype(df,col_list,_type=int):
    for col in col_list:
      df[col]=df[col].astype(_type)
    return df

def to_num(df,col_list):
    for col in col_list:
        df[col]=pd.to_numeric(df[col], errors='coerce')
        return df
    
def to_datetime(df,col_list):
    for col in col_list:
        df[col]=pd.to_datetime(df[col])
    df['count_day']=(df['VSD']-df['Order date']).dt.days
    df['month_order']=df['Order date'].dt.month
    df['day_order']=df['Order date'].dt.day

    df['day_receive']=df['VSD'].dt.day
    df['month_receive']=df['VSD'].dt.month
    df['year_receive']=df['VSD'].dt.year
    return df

def fill_weight_unit(df):
    df['WEIGHT_UNIT']=df['WEIGHT_UNIT'].fillna(2)
    df['WEIGHT_UNIT']=df['WEIGHT_UNIT'].replace('g',1)
    df['weight']=df['WEIGHT_UNIT']*df['WEIGHT PER PIECE']*df['SO QTY']
    return df


if __name__ == '__main__':
    train = pd.read_csv('./train.csv')
    train = train.replace(r'^\s*$',0, regex=True)
    test = pd.read_csv('./test.csv')
    test = test.replace(r'^\s*$',0, regex=True)

    list_col_to_datetime=['Order date','VSD']
    list_col_to_num=['Consider count hodiday Saturday','OTHER AREA SHIP DIV']
    list_astype=['OTHER AREA SHIP DIV']
    list_col_drop=['ALLOCATION QTY','CUST_CD','GLOBAL_NO','HAZARD_FLG','HEAVY_FLG',
                   'INNER_CD','IO_UNFIT_FLG','Order date','OTHER AREA SHIP DIV',
                   'PACKING RANK','PRODUCT ATTRIBUTION','PRODUCT_ASSORT','PRODUCT_CD',
                   'PURCHASE AMOUNT','QTUF_RCV_NO','Ship Mode','SO_DAY_OF_MONTH','SO_TIME',
                   'SOUF_RCV_NO','SPECIAL DIV','SPECIFY_PRODUCTION_DAYS','SPECIFY_SHIP_DAYS',
                   'SUBSIDIARY_CD','SUPPLIER INV AMOUNT','WEIGHT PER PIECE','WEIGHT_UNIT']
    #clist_col_drop=['GLOBAL_NO','SOUF_RCV_NO','QTUF_RCV_NO','SUBSIDIARY_CD','PRODUCT_ASSORT','HAZARD_FLG','PRODUCT_CD']
    #xử lý train
    train=fill_weight_unit(train)
    train=fill_missing(train)
    train=to_datetime(train,list_col_to_datetime)
    train=to_num(train,list_col_to_num)
    train=astype(train,list_astype)
    train=drop_col(train,list_col_drop)
    train,la_encoder=label_encoder(train)
    #train=astype(train,train.columns.to_list()[1:],'category')
    train.to_pickle('train.pkl')
    #xử lý test
    test=fill_weight_unit(test)
    test=fill_missing(test)
    test=to_datetime(test,list_col_to_datetime)
    test=to_num(test,list_col_to_num)
    test=astype(test,list_astype)
    test=drop_col(test,list_col_drop)
    test=label_encoder_test(test,la_encoder)
    test=fill_missing(test)
    #test=astype(test,test.columns.to_list()[1:],'category')
    test.to_pickle('test.pkl')
    print(train.info())
    print(test.info())