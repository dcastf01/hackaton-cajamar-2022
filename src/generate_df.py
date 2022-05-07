import pandas as pd

from src.utils import applyParallel_using_df_standar,modify_path_if_week_periocity
import os
import logging
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
import multiprocessing
from joblib import Parallel, delayed
def get_raw_data(data_path:str=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\01_raw\Modelar_UH2022.txt'):
        df=pd.read_csv( data_path,sep="|")
        # df=df[df['ID']<=25]
        # df=df[df['ID'].isin(list_of_id_with_tiny_data)]
        logging.info(df.shape)
        df.drop(columns=['READINGINTEGER','READINGTHOUSANDTH'],inplace=True)
        df.fillna(0,inplace=True) #check in future
        logging.info(df.shape)
        df.loc[:,'SAMPLETIME']=pd.to_datetime(df['SAMPLETIME'],errors='raise',format='%Y-%m-%d')
        return df

def generate_df(
    data_path:str=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\01_raw\Modelar_UH2022.txt',   
    D_or_W:str='D',
    df_to_choose_model:bool=False,
    ):

    #df_to_choose_model only check outliers until 24 of january
    path_dates_check=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\03_primary\df_clean_wiht_all_dates.csv'
    if df_to_choose_model:
        suffixes='df_final.csv'
        path_dates_check=path_dates_check.split('.')[0]+suffixes
    path_dates_check=modify_path_if_week_periocity(path_dates_check,D_or_W)
    if os.path.isfile(path_dates_check):
       df_total_without_outliers=pd.read_csv(path_dates_check)
       
    else:
        logging.info('generating df')
        df_raw=get_raw_data(data_path)
        logging.info(df_raw.shape)
        logging.info('cleaning raw df')
        df=clean_data(df_raw)    
        logging.info(df.shape)
        df.rename(columns={'SAMPLETIME':'date'},inplace=True)
        df['total']=df['DELTAINTEGER']+df['DELTATHOUSANDTH']/100
        df.drop(['DELTAINTEGER','DELTATHOUSANDTH'],axis=1,inplace=True)

        # df_total=pd.DataFrame()
        kwargs={'D_or_W': D_or_W}
        df_total=applyParallel_using_df_standar(df,checking_date_df,'checking dates',njobs=11,**kwargs)
        logging.info('cleaning df grouped')
        df_total_without_outliers=outliers_after_groupby_period(df_total,period=D_or_W)
        logging.info(df.shape)
        df_total_without_outliers=df_total_without_outliers.reset_index()
        df_total_without_outliers.to_csv(path_dates_check,index=False)
    # df_total=df_total[df_total['ID']<=10]
    return df_total_without_outliers
def negative_value_to_0(df):
        df['DELTAINTEGER']=np.where(df['DELTAINTEGER']<0,0,df['DELTAINTEGER'])
        return df
def negative_value_to_mean(df):
        df['DELTAINTEGER']=np.where(df['DELTAINTEGER']<0,df['DELTAINTEGER'].mean(),df['DELTAINTEGER'])
        return df    
def check_if_we_have_negative(df,column='DELTAINTEGER'):
    return df[df[column] < 0][column].any()
def clean_data(df):
    def clean_instances_with_minor_error(df,special_unique_negative_id):
            
        negative_instances=df[df['DELTAINTEGER']<0].sort_values(['ID','SAMPLETIME'])
        normal_unique_negative=list(negative_instances[~negative_instances.isin(special_unique_negative_id)]['ID'].unique())
        # print(df[df['ID'].isin(normal_unique_negative)].describe())
        df_normal_unique_negative=pd.DataFrame()
        for id_unique in tqdm(normal_unique_negative):
            df_per_id=df[df['ID']==id_unique]
            df_per_id=negative_value_to_0(df_per_id)
            df_per_id["ID"]
            df_normal_unique_negative=pd.concat([df_normal_unique_negative,df_per_id])#.reset_index()
        # print(df_normal_unique_negative.describe())
        return df_normal_unique_negative

    def clean_outliers_with_diverse_strategy(df,id_outliers_with_a_fews_peaks:dict):
        
        df_outlier_unique_negative=pd.DataFrame()
        for id_unique,args in tqdm(id_outliers_with_a_fews_peaks.items()):
            df_outlier_strategy_per_id=df[df['ID']==id_unique].sort_values(['ID','SAMPLETIME'])   
            # print(df_outlier_strategy_per_id.describe())     
            df_outlier_strategy_per_id['DELTAINTEGER']=np.where(   
                    df_outlier_strategy_per_id['DELTAINTEGER']<0,
                    df_outlier_strategy_per_id[(df_outlier_strategy_per_id['DELTAINTEGER']>0)&(df_outlier_strategy_per_id['DELTAINTEGER']<args['limit_max'])]['DELTAINTEGER'].mean(),
                    np.where(
                        df_outlier_strategy_per_id['DELTAINTEGER']>args['limit_max'],
                            args['limit_max']+0.2,
                            df_outlier_strategy_per_id['DELTAINTEGER']
                    )
                    ) 
            # print(df_outlier_strategy_per_id.describe()) 
            exist_negative=check_if_we_have_negative(df_outlier_strategy_per_id)
            if exist_negative:
                df_outlier_strategy_per_id=negative_value_to_mean(df_outlier_strategy_per_id)
            df_outlier_unique_negative=pd.concat([df_outlier_unique_negative,df_outlier_strategy_per_id])
        # print(df_outlier_unique_negative.describe())
        return df_outlier_unique_negative

    def strategy_inverse_to_clean_data(df,id_to_inverse_negative_values:list):

        df_inverse_strategy_unique_negative=pd.DataFrame()
        for id_unique in tqdm(id_to_inverse_negative_values):

            df_inverse_strategy_per_id=df[df['ID']==id_unique].sort_values(['ID','SAMPLETIME'])
            df_inverse_strategy_per_id['DELTAINTEGER']=abs(df_inverse_strategy_per_id['DELTAINTEGER'])

            exist_negative=check_if_we_have_negative(df_inverse_strategy_per_id)
            if exist_negative:
                print('ID ',id_unique,'already with negatives')
                df_inverse_strategy_per_id=negative_value_to_mean(df_inverse_strategy_per_id)
            df_inverse_strategy_unique_negative=pd.concat([df_inverse_strategy_unique_negative,df_inverse_strategy_per_id])
        return df_inverse_strategy_unique_negative

    special_unique_negative_id=[57, 301,374,379,493,635,845,
                            873, 907,1218,1280,1468,1487,
                            1506,1739,1873,1884, 2063,2121,
                            2711
                            ] #obtenidos del estudio exploratorio
    
    id_to_inverse_negative_values=[
                                    2121,
                                    2711,
                                    1873,
                                    379,
                                    1739,
                                    1468,]
    
    id_outliers_with_a_fews_peaks={
        57:{'limit_max':100},
        374:{'limit_max':2700},
        493:{'limit_max':200},
        635:{'limit_max':200},
        873:{'limit_max':200},
        907:{'limit_max':50},
        1218:{'limit_max':400},
        1280:{'limit_max':250},
        2063:{'limit_max':300},
        1884:{'limit_max':150},
        1506:{'limit_max':200},
        1487: { 'limit_max':150},
        301:{ 'limit_max':150},
        # cuando se pase esto a dias poner la media en ese fragmento que desaparece de este ID el id 845:
        #valores entre el 1 de mayo y el 8 de junio de 2019 se tienen que corregir de forma diaria
        845:{ 'limit_max':250},    
        }
    
    ids_with_negatives=df[df['DELTAINTEGER']<0].sort_values(['ID','SAMPLETIME'])['ID'].unique()
    df_with_negatives=df[df["ID"].isin(ids_with_negatives)].copy()
    df_without_negatives=df[~df["ID"].isin(ids_with_negatives)].copy() #pendiente ver los picos máximos para solucionarlo
    df_normal_unique_negative=clean_instances_with_minor_error(df_with_negatives,special_unique_negative_id)
    df_outlier_unique_negative=clean_outliers_with_diverse_strategy(df_with_negatives,id_outliers_with_a_fews_peaks)
    df_inverse_strategy_unique_negative=strategy_inverse_to_clean_data(df_with_negatives,id_to_inverse_negative_values)
    df_total=pd.concat([df_without_negatives,df_normal_unique_negative,df_outlier_unique_negative,df_inverse_strategy_unique_negative])
    
    return df_total.sort_values(['ID','SAMPLETIME'])
    
def checking_date_df(df_unique_id,id_unique,D_or_W='D'):
    def get_range_to_predic(start='2020-01-31',last_date='2020-01-31',periods=1,D_or_W='D'):
       
        freq=D_or_W
        dates = pd.date_range(
            start=start,
            periods=periods,  # An extra in case we include start
            freq=freq)
    
        return dates
    def add_new_date_per_id(df_per_id_but_without_column_id,date,id_unique):
        df_to_add_new_date=pd.DataFrame()
    
        df_to_add_new_date.index=get_range_to_predic(start=date,last_date=date)
        df_to_add_new_date.index = df_to_add_new_date.index.set_names(['date'])
        df_to_add_new_date.reset_index(inplace=True)
        
        df_to_add_new_date=df_to_add_new_date[df_to_add_new_date['date']==date]
        
        return df_to_add_new_date
    df_per_id=df_unique_id
    df_per_id_but_without_column_id=df_per_id.drop('ID',axis=1)
    list_of_df_with_new_dates=[]
    date='2020-01-31'
    if not df_per_id.date.isin([date]).any():
        # print('adding',date)
        df_with_last_date=add_new_date_per_id(df_per_id_but_without_column_id,date,id_unique)
        list_of_df_with_new_dates.append(df_with_last_date)
    # else:print(id_unique,'ya tiene la fecha',date)
    date='2019-02-01'
    if not df_per_id.date.isin([date]).any():
        # print('adding',date)
        df_with_start_date=add_new_date_per_id(df_per_id_but_without_column_id,date,id_unique)
        list_of_df_with_new_dates.append(df_with_start_date)
    # else:print(id_unique,'ya tiene la fecha',date)
    if list_of_df_with_new_dates:
        # print('list_of_df_with_new_dates')
        df_aux=pd.concat([df_per_id_but_without_column_id,*list_of_df_with_new_dates])
    else:    
        df_aux=df_per_id_but_without_column_id
        # print('no_new_date')
    # df_total.drop('ID',inplace=True,axis=1)
    if D_or_W=='W':
        resample_freq='W-SAT' #debería de ser W-FRI me equivoque en un día
    elif  D_or_W=='D':
        resample_freq=D_or_W
    df_aux=df_aux.resample(resample_freq,on='date').sum()
    df_aux['ID']=id_unique
    return df_aux

def outliers_after_groupby_period(df,outliers_fraction=0.01,period='D',njobs=11):
    def clean_outliers(df_per_id,id_unique,outliers_fraction=0.01,period='D'):

        if period=='D':
            periods=7
        elif period=='W':
            periods=1
        else:
            raise 'Option period no valid'

        method_forest=IsolationForest(contamination=outliers_fraction)
        yhat = method_forest.fit_predict(df_per_id['total'].to_numpy().reshape(-1,1))
        mask = yhat != -1
        df_per_id['total']=np.where((mask) | (df_per_id.index> '2020-01-22'),
                                    df_per_id['total'],
                                    df_per_id['total'].shift(periods=periods).fillna(method='bfill')
                            )
        df_per_id['ID']=id_unique
        return df_per_id
        # df_after_outliers=pd.concat([df_after_outliers,df_per_id])
    
    df_after_outliers=pd.DataFrame()
    desc='checking outliers and removing'
    if njobs==-1 :n_jobs=multiprocessing.cpu_count()  
    
    retLst = Parallel(n_jobs=njobs)(delayed(clean_outliers)(
        df[df['ID']==id],id,outliers_fraction,period) for id in tqdm(df.ID.unique(),
                                                                    desc=desc,
                                                                    mininterval=25))
    df_after_outliers=pd.concat(retLst)    
    return df_after_outliers