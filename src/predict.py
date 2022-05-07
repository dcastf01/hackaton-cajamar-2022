
import json
import logging
import os
from cmath import exp

import numpy as np
import pandas as pd
from prophet.serialize import model_from_json, model_to_json
from pycaret import regression
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from src.create_features import create_features
from src.utils import get_range_of_dates,transform_df_in_pivot_with_id_in_columns,modify_path_if_week_periocity


def predict_per_id(df_with_features:pd.DataFrame,df_summary_errors:pd.DataFrame,D_or_W):
    def select_model_per_id_and_type_model(model_choose,id_unique=None):
        root_models=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models\last_model_per_id'
        # df_summary_errors_aux=df_summary_errors[df_summary_errors['ID']==id_unique]
        # model_choose=df_summary_errors_aux['best_model'].values[0]
        if  'pycaret_all_data' in model_choose:
            extension_model=''
            which_model=str(model_choose.split("_")[-1])
            name_model=f'best_model_all_data_{which_model}{extension_model}'
            path_to_model=modify_path_if_week_periocity(root_models, D_or_W)
            path_to_this_model=os.path.join(path_to_model,name_model)
            type_prediction='pycaret_regression'
            model=regression.load_model(path_to_this_model)
        elif  'Prophet' in model_choose :
            extension_model='.json'
            name_model=f'train_without_exogenous_variables_serialized_model_id_{id_unique}{extension_model}'
            path_to_model=modify_path_if_week_periocity(root_models, D_or_W)
            path_to_this_model=os.path.join(path_to_model,name_model)
            type_prediction='prophet'
            with open(path_to_this_model, 'r') as fin:
                    model = model_from_json(json.load(fin))

        elif  'clusters_mean' in model_choose :
            extension_model=''
            which_model_raw=model_choose.split("_")
            which_model='_'.join(which_model_raw[-2:])
            name_model=f'best_model_per_clusters_mean_group_{which_model}{extension_model}'
            path_to_model=modify_path_if_week_periocity(root_models, D_or_W)
            path_to_this_model=os.path.join(path_to_model,name_model)
            type_prediction='pycaret_regression'
            model=regression.load_model(path_to_this_model)
        elif  'clusters_std' in model_choose :
            extension_model=''
            which_model_raw=model_choose.split("_")
            which_model='_'.join(which_model_raw[-2:])
            name_model=f'best_model_per_clusters_std_group_{which_model}{extension_model}'
            path_to_model=modify_path_if_week_periocity(root_models, D_or_W)
            path_to_this_model=os.path.join(path_to_model,name_model)
            type_prediction='pycaret_regression'
            model=regression.load_model(path_to_this_model)

        return model,type_prediction
    def prophet_prediction_agroupation(df,id_unique,kind_model,D_or_W):
        df_aux=df[df['ID']==id_unique]
        model,type_prediction=select_model_per_id_and_type_model(kind_model,id_unique)
        predictions_per_id=predict(model,df_aux,start_date='2020-02-01',D_or_W=D_or_W,type_prediction=type_prediction)
        return predictions_per_id

    predictions=pd.DataFrame()
    predictions=pd.read_csv(r'D:\programacion\Repositorios\datathon-cajamar-2022\prediction_backup.csv')
    print(predictions.head())
    ids_ya_hechos=predictions.ID.unique()
    print(predictions.ID.nunique())
    print(df_summary_errors.ID.nunique())
    print(df_with_features.ID.nunique())
    for kind_model in tqdm(df_summary_errors['best_model'].unique(),desc='Prediciendo por tipo de modelo'):
        #se podrían agrupar por modelos y luego lanzar
        ids=df_summary_errors[df_summary_errors['best_model']==kind_model].ID.unique()
        df_aux=df_with_features[df_with_features['ID'].isin(ids)]
        df_aux=df_aux[~df_aux['ID'].isin(ids_ya_hechos)]
        if not df_aux.empty:
            print(df_aux.head())
            if kind_model=='Prophet_without_exogenous':
                desc='Prediciendo los de prophet'
                outputs = Parallel(n_jobs=12)(delayed(prophet_prediction_agroupation)(
                        df_aux,id_unique,kind_model,D_or_W) for id_unique in tqdm(df_aux.ID.unique(),
                                                            desc=desc,
                                                            mininterval=25)
                                                            )
                predictions=pd.concat([outputs,predictions])
                predictions.to_csv('prediction_backup.csv',index=False)
                # for id_unique in  tqdm(df_aux.ID.unique(),desc='Prediciendo los de prophet'):
                    
                #     df_aux=df_with_features[df_with_features['ID']==id_unique]
                #     model,type_prediction=select_model_per_id_and_type_model(kind_model,id_unique)
                #     predictions_per_id=predict(model,df_aux,start_date='2020-02-01',D_or_W=D_or_W,type_prediction=type_prediction)
                #     predictions=pd.concat([predictions,predictions_per_id])
            else:
                id_unique=None
                model,type_prediction=select_model_per_id_and_type_model(kind_model,id_unique)
                predictions_per_id=predict(model,df_aux,start_date='2020-02-01',D_or_W=D_or_W,type_prediction=type_prediction)
                predictions=pd.concat([predictions,predictions_per_id])
                predictions.to_csv('prediction_backup.csv',index=False)
        print(predictions.head())
    return predictions.sort_values(['ID','date'])

def predict(model,df,start_date:str='2020-02-01',D_or_W='D',type_prediction='pycaret_regression'):
    if type_prediction=='pycaret_regression':
        predictions=predict_pycaret_regression(model,df,start_date,D_or_W)
    elif type_prediction=='prophet':
        predictions=prediction_prophet(model,df,D_or_W)
    return predictions


def prediction_prophet(model,df,D_or_W):
    def do_prediction(model,period=7):
        future=model.make_future_dataframe(period)
        forecast = model.predict(future)
        #avoid negative numbers
        forecast['yhat']=np.where(forecast['yhat']<0,
                                0,
                                forecast['yhat'])
        
        return forecast.tail(period)
    if D_or_W=='D':
        period=7
    elif D_or_W=='W':
        period=2
    predictions_output=pd.DataFrame()
    pivot_by_dates=transform_df_in_pivot_with_id_in_columns(df)
    id_uniques=list(df.ID.unique())
    if len(id_uniques)==1:
        id_unique=id_uniques[0]
    else:
        raise 'error en los id prediciendo el prophet'

    df_prophet=pd.DataFrame()
    df_prophet[['ds','y']]=pivot_by_dates.reset_index()
    predictions=do_prediction(model,period)
    predictions_output[['date','Label']]=predictions[['ds','yhat']]
    predictions_output['total']=predictions_output['Label']
    predictions_output['ID']=id_unique

    return predictions_output

def predict_pycaret_regression(model,df,start_date:str='2020-02-01',D_or_W='D'):
    #we need predict 14 days of February 7 days to predict and 2 weeks
    
    df['date']=pd.to_datetime(df['date'])
    # df=df[df['ID']<=10]
    # df=df[df['date']>=pd.to_datetime('2019-09-15')]
    # model=regression.load_model(model)
    if D_or_W=='D':
        periods=7
    elif D_or_W=='W':
        periods=2
    else:
        raise 'error en predict'
    dates=get_range_of_dates(start_date,periods,D_or_W=D_or_W)
    unique_id=df.ID.unique()
    predictions=pd.DataFrame()
    # print(df.ID.nunique())
    for day_to_predict in tqdm(dates,desc='calculando predicciones por día'):
        print(day_to_predict)
        df_aux=pd.DataFrame()
        for id_unique in unique_id:
            df_per_id=pd.DataFrame()
            df_per_id.index=dates
            df_per_id.index = df_per_id.index.set_names(['date'])
            df_per_id.reset_index(inplace=True)
            df_per_id['ID']=id_unique

            df_per_id=df_per_id[df_per_id['date']==day_to_predict]

            df_aux=pd.concat([df_aux,df_per_id])
        # df_aux.loc[:,'date']=pd.to_datetime(df_aux['date'],errors='raise',format='%Y-%m-%d')
        # print(df_aux.head())
        df_to_predict=df_aux
        # print(df_to_predict.head())
        df_full_to_get_temporal_features=pd.concat([df_to_predict,df])
        df_full_to_get_temporal_features.sort_values(['ID','date'],inplace=True)
        df_full_to_get_temporal_features.reset_index(drop=True,inplace=True)
        # print(df_full_to_get_temporal_features[df_full_to_get_temporal_features['ID']==2].head())
        # print(df_full_to_get_temporal_features[df_full_to_get_temporal_features['ID']==2].tail())
        # print(df_full_to_get_temporal_features.date.max())
        # print(df_full_to_get_temporal_features.date.min())
        #to reduce compute cost we don't need data more than 90 days
        # df_full_to_get_temporal_features['total']=df_full_to_get_temporal_features['total'].fillna(0)
        auxiliar=df_full_to_get_temporal_features[df_full_to_get_temporal_features['date']>=pd.to_datetime('2020-01-29')].sort_values('ID')
        # print(auxiliar.columns) 
        # print(auxiliar.head(40))
        logging.debug('creating features')
        df_full=create_features(df_full_to_get_temporal_features[['ID','total','date']],is_to_predict=True,D_or_W=D_or_W)
        auxiliar=df_full[df_full['date']>=pd.to_datetime('2020-01-30')].sort_values('ID')
        # print(auxiliar.head(40))
        logging.debug('we have the features')
        df_to_predict=df_full[df_full['date']==pd.to_datetime(day_to_predict)]
        logging.debug(df_to_predict.shape)
        results=pd.DataFrame()
        # print(df_to_predict.head(20))
        # df_to_predict.drop('total',axis=1,inplace=True)
        try:
            results=regression.predict_model(model,df_to_predict)
            results['Label']=np.where(results['Label']<0,
                                0,
                                results['Label'])
        except Exception as e:
            print(e) #debug purpose
        predictions=pd.concat([predictions,results])
        # print(predictions.ID.nunique())
        # print(results.shape)
        # results['total']=0
        # print(results[['ID','total','date','Label']].head(30))
        # print(results[results['date']==pd.to_datetime(day_to_predict)][['ID','total','date']].tail(5))
        results['total']=results['Label']
        
        # print(results[['ID','total','date','Label']].head(30))
        results.drop('Label',axis=1,inplace=True)
        # print(df.shape)
        df=pd.concat([df,results])
        # print(df.shape)
        
        df.sort_values(['ID','date'],inplace=True)
        df.reset_index(drop=True,inplace=True)
    # predictions=df
    predictions['total']=predictions['Label']
    predictions.to_csv('last_prediction.csv',index=False)


    return predictions[['ID','Label','date','total']]
