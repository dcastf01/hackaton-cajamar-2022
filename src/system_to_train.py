
from genericpath import exists
import json
import logging
import multiprocessing
import os
import warnings
import re
from typing import Optional, Tuple, Union
from datetime import datetime
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from pycaret import regression, time_series
from tqdm import tqdm
from src.predict import predict
from src.utils import (applyParallel_using_df_standar,
                       applyParallel_using_pivot_table, get_range_of_dates,
                       rmse, stan_init,modify_path_if_week_periocity,
                       transform_df_in_pivot_with_id_in_columns)
import shutil
warnings.simplefilter(action='ignore', category=FutureWarning)

import yaml


def get_pycaret_attributes():
    with open(r"D:\programacion\Repositorios\datathon-cajamar-2022\src\pycaret_attributes.yml", "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def split_data(data, split_date='2020-01-24',column_date=None):
    if column_date:
        return data[data[column_date] <= split_date].copy(), \
           data[data[column_date] >  split_date].copy()
           
    else: return data[data.index <= split_date].copy(), \
           data[data.index >  split_date].copy()

def split_data_pivot_table(pivot_by_dates,ts,split_date='2020-01-24'):
    return split_data(pivot_by_dates[ts],split_date )
     
def train_systems_with_prophet_without_exogenous(df,njobs=11,D_or_W='D'):
    def do_prediction(model,period=7,D_or_W='D'):
        freq=D_or_W
        if D_or_W=='D':
            period=7
        elif D_or_W=='W':
            period=2
        future=model.make_future_dataframe(period,freq=freq)
        forecast = model.predict(future)
        #avoid negative numbers
        forecast['yhat']=np.where(forecast['yhat']<0,
                                0,
                                forecast['yhat'])
        
        return forecast.tail(period)
    def prophet_without_exogenous_variables(train,test,ts,D_or_W='D') :
        print('id',ts)
        root_path_model=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models\prophet'
        root_path_model=modify_path_if_week_periocity(root_path_model, D_or_W)
        os.makedirs(root_path_model,exist_ok=True)
        fpath_model=os.path.join(root_path_model,f'train_without_exogenous_variables_serialized_model_id_{ts}.json')
        error_df=pd.DataFrame()
        df_prophet=pd.DataFrame()
        df_prophet[['ds','y']]=train.reset_index()
        if D_or_W=='D':
            weekly_seasonality=True
        else:
            weekly_seasonality=False
        if os.path.exists(fpath_model):
            try:
                with open(fpath_model, 'r') as fin:
                    m_prev = model_from_json(json.load(fin))  # Load model   
                m = Prophet(yearly_seasonality=False,
                        weekly_seasonality=weekly_seasonality,
                        daily_seasonality=False,
                        )
            
                m.fit(df_prophet,init=stan_init(m_prev))
            except Exception as e:
                logging.error('una excepción donde prophet')
                logging.error(e)
                m = Prophet(yearly_seasonality=False,
                weekly_seasonality=weekly_seasonality,
                daily_seasonality=False
                )
                m.fit(df_prophet)
                with open(fpath_model, 'w') as fout:
                    json.dump(model_to_json(m), fout)  # Save model
        else:
                       
            m = Prophet(yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=False
                )
            m.fit(df_prophet)
            with open(fpath_model, 'w') as fout:
                json.dump(model_to_json(m), fout)  # Save model
      
        prediction=do_prediction(m,D_or_W=D_or_W) #queda pendiene cambiar esto a predict para que así no le importe si es semanal o diario
        error=rmse(test,prediction.yhat)
        error_df.loc[ts,'Prophet_without_exogenous']=error
        error_df.index.name='ID'
        prediction['ID']=ts
        return error_df,prediction
    
    path_prophet_errors=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting\errors\errors_prophet_without_exogenous_variables.csv'
    path_prophet_errors=modify_path_if_week_periocity(path_prophet_errors, D_or_W)
    path_prophet_predictions=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\07_model_outputs\predictions_prophet_without_exogenous_variables.csv'
    path_prophet_predictions=modify_path_if_week_periocity(path_prophet_predictions, D_or_W)
 
    if os.path.isfile(path_prophet_errors) and os.path.isfile(path_prophet_predictions):
        errors_df=pd.read_csv(path_prophet_errors)
        predictions_df=pd.read_csv(path_prophet_predictions)
    else:
        # df=df[df['ID']<10] #to debug
        pivot_by_dates=transform_df_in_pivot_with_id_in_columns(df)
        pivot_by_dates_train,pivot_by_dates_test=split_data(pivot_by_dates)

        desc='entrenando prophet sin variables exogenous'
        ts=11
        print(pivot_by_dates.columns)
        b=prophet_without_exogenous_variables(pivot_by_dates_train[ts],pivot_by_dates_test[ts],ts,D_or_W=D_or_W)

        # for ts in tqdm(pivot_by_dates.columns,desc=desc,mininterval=25):
        #     try:
        #         prophet_without_exogenous_variables(pivot_by_dates_train[ts],pivot_by_dates_test[ts],ts,D_or_W)
        #     except Exception as e:
        #         print(e)
        #         print(ts)
        outputs = Parallel(n_jobs=njobs)(delayed(prophet_without_exogenous_variables)(
            pivot_by_dates_train[ts],pivot_by_dates_test[ts],ts,D_or_W) for ts in tqdm(pivot_by_dates.columns,
                                                        desc=desc,
                                                        mininterval=25)
                                                        )
        errors = [item[0] for item in outputs]
        predictions = [item[1] for item in outputs]
        errors_df=pd.concat(errors)
        predictions_df_with_all_variables=pd.concat(predictions)
        predictions_df=pd.DataFrame()
        predictions_df[['date','total','ID']]=predictions_df_with_all_variables[['ds','yhat','ID']]
        print(errors_df.head())
        print(predictions_df.head())  
        errors_df.to_csv( path_prophet_errors)
        predictions_df.to_csv(path_prophet_predictions,index=False)
            
    return errors_df,predictions_df

def train_systems_with_pycaret_regression(df,njobs=11,D_or_W='D'):
    def do_prediction(model,period=7):
        predictions=time_series.predict_model(model,fh=7,return_pred_int=False,round=2) 
        predictions=np.where(predictions<0,
                                0,
                                predictions)
        return predictions
    def pycaret_without_exogenous_variables(train,test,ts,period=7,D_or_W='D') :
        root_path_model=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models\pycaret_timeseries'
        root_path_model=modify_path_if_week_periocity(root_path_model, D_or_W)
        os.makedirs(root_path_model,exist_ok=True)
        fpath_model=os.path.join(root_path_model,f'train_without_exogenous_variables_serialized_model_id_{ts}')
        error_df=pd.DataFrame()       
        fh=period
        if ts==151 :
            print('stop')
        try:
            if os.path.exists(fpath_model+'.pkl'):
                best=time_series.load_model(fpath_model)
            else:                      
                time_series.setup(
                    data=train, fh=fh,
                    enforce_exogenous=False,
                    # Set defaults for the plots ----
                    # fig_kwargs={"renderer": "notebook", "width": 1000, "height": 600},
                    session_id=42,
                    verbose=False,
                    use_gpu=True
                    # system_log=False,
                    )
                best=time_series.compare_models(turbo=True,sort='RMSE',round=2,
                                                    exclude=[
                                                            'huber_cds_dt',
                                                            'tbats',
                                                            'bats',
                                                            'auto_arima',
                                                            'prophet'
                                                            ],
                                                            # errors='raise',
                                                            )

                best=time_series.finalize_model(best)
                time_series.save_model(best,fpath_model) 
            predictions=do_prediction(best,period)
            predictions_df=pd.DataFrame()
            predictions_df['total']=predictions
            dates=get_range_of_dates('2020-01-24',periods=7)
            error=rmse(test,predictions)
            error_df.loc[ts,'Pycaret_time_series_without_exogenous']=error
            error_df.index.name='ID'
            predictions_df['ID']=ts
            predictions_df['date']=dates
            return error_df,predictions_df
        except Exception as e:

            print(e)
            X_train=time_series.get_config('X_train')
            print(X_train)
            print(time_series.pull())
            print(train.describe())
            print(test.describe())
            print('error')
    
    path_pycaret_regresion_errors=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting\errors\errors_pycaret_time_series_without_exogenous_variables.csv'
    path_pycaret_regresion_predictions=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\07_model_outputs\predictions_pycaret_time_series.csv'
    
    if os.path.isfile(path_pycaret_regresion_errors) and os.path.isfile(path_pycaret_regresion_predictions):
        errors_df=pd.read_csv(path_pycaret_regresion_errors)
        predictions=pd.read_csv(path_pycaret_regresion_predictions)
    else:
        # df=df[df['ID']<10] #to debug
        print(df.dtypes)
        pivot_by_dates=transform_df_in_pivot_with_id_in_columns(df)
        print(pivot_by_dates.dtypes)
        pivot_by_dates_train,pivot_by_dates_test=split_data(pivot_by_dates)
        print(pivot_by_dates_train.dtypes)
        desc='entrenando pycaret time regression sin variables exogenous'
        ts=0
        # 
        # a=pycaret_without_exogenous_variables(pivot_by_dates_train[ts],pivot_by_dates_test[ts],ts)
        try:
            outputs = Parallel(n_jobs=njobs)(delayed(pycaret_without_exogenous_variables)(
                pivot_by_dates_train[ts],pivot_by_dates_test[ts],ts) for ts in tqdm(pivot_by_dates.columns,
                                                            desc=desc,
                                                            mininterval=25))
        except:
            for ts in tqdm(pivot_by_dates.columns,desc=desc,mininterval=25):
                outputs=pycaret_without_exogenous_variables(pivot_by_dates_train[ts],pivot_by_dates_test[ts],ts)
        errors = [item[0] for item in outputs]
        predictions = [item[1] for item in outputs]
        errors_df=pd.concat(errors)
        predictions_df=pd.concat(predictions)
        print(errors_df.head())
        print(predictions_df.head())  
        errors_df.to_csv( path_pycaret_regresion_errors,index=False)
        predictions_df.to_csv(path_pycaret_regresion_predictions,index=False)
            
    return errors_df,predictions       

def setup_pycaret(df_train:pd.DataFrame,pycaret_atributtes:dict):
    """
    This function create the setup to pycaret to later can use the different functions of pycaret

    Args:
        data (Tuple[pd.DataFrame,dict]): Tuple with information from master table
            and dictionary with extra information everything in catalog.yml
        pycaret_atributtes (dict): dictionary from parameters.yml with values to create the models and setup from pycaret
    """        
    
    target_column=pycaret_atributtes["target_column"]
        
    print(df_train.dtypes)
    logging.info(f"data to train shape:{df_train.shape}")
    print(df_train.head())
    print(df_train.dtypes)
    # logging.info(f' we have this folds{df_train["year"].unique()}')
    # print(df.isna().sum())
    
    regression.setup(
        data=df_train,
        target=target_column,
        # test_data=df_test,    
        numeric_imputation = 'mean',
        fold_strategy='kfold',
        ignore_features=['ID','date',],
        #faltará añadir más datos en numericos de estacionalidd, tendencia, etc por cada ID mediante prophet
            categorical_features = [
                # 'dayofweek',
                # 'quarter','month',
                #  'year',
                    # 'dayofyear',
                    # 'dayofmonth',
                    # 'weekofyear','ID'
                    ]  , 
        normalize=True,
        transformation = True,
        transform_target = False, 
        # transform_target_method='yeo-johnson',
        combine_rare_levels = True, rare_level_threshold = 0.1,
        # remove_multicollinearity = True, multicollinearity_threshold = 0.95, 
        # feature_selection=True,
        # create_clusters=True,
            silent = True,
        fold_shuffle=False,
        n_jobs=10,
        # shuffle=True    
            )

    regression.set_config("seed",333)
    # X_train=regression.get_config('X_train')
    # print(X_train.head())
    # print(X_train.shape)

def run_pycaret(df_train:pd.DataFrame,)->Tuple[pd.DataFrame,pd.DataFrame,list]: 
    """
    We create the setup and compare differents models with the high-level API of pycaret

    Args:
        df pd.DataFrame): Tuple with information from master table
            and dictionary with extra information everything in catalog.yml
        pycaret_atributtes (dict): dictionary from parameters.yml with values to create the models and setup from pycaret
    Returns:
        Tuple[pd.DataFrame,pd.DataFrame,list]: _description_
    """  
    pycaret_attributes=get_pycaret_attributes()
    def train_the_model_with_specific_type_of_train_and_get_results(
        models:dict,
        func:str,
        results:Optional[pd.DataFrame]=None,
        suffix_to_model_name:Optional[str]=None,
        **kwargs_model:Optional[dict],
        )->Tuple[dict,pd.DataFrame]:
        """
        _summary_

        Args:
            models (dict): _description_
            func (str): _description_
            results (Optional[pd.DataFrame], optional): _description_. Defaults to None.
            suffix_to_model_name (Optional[str], optional): _description_. Defaults to None.

        Returns:
            Tuple[dict,pd.DataFrame]: _description_
        """        
        
        if results is None:  
            results=pd.DataFrame()
        method=getattr(regression,func) 
        new_models={}
        
        for model_name,model in models.items():
            logging.info(f'creating the model: {model}')
            try:
                model_trained=method(model,**kwargs_model)
                logging.info(f'fitted model: {model_trained}')
                model_name=add_suffix_to_model_name(model_name,model_trained,suffix_to_model_name)
                result=create_normalized_df_with_cv_after_create_model(model_name)
                results=results.append(result,ignore_index=True)
                new_models[model_name]=model_trained
            except Exception as e:
                logging.error(e)
            
        models={**models,**new_models}
        results.reset_index(inplace=True,drop=True)
        return models,results
    
    def create_models_and_get_results(models:dict,
                                      results:Optional[pd.DataFrame]=None,
                                      suffix_to_model_name:Optional[str]=None)->Tuple[dict,pd.DataFrame]:
        """Function that create again the models in topn_models to have results per fold in cross validation
        Args:
            topn_models (list): List with top n models
            results (Optional[pd.DataFrame], optional): [description]. Defaults to None.
            suffix_to_model_name (Optional[str], optional): [description]. Defaults to None.

        Returns:
            tuple: first value are the models and the second the dataframe results
        """        
        return train_the_model_with_specific_type_of_train_and_get_results(
            models=models,
            func="create_model",
            results=results,
            suffix_to_model_name=suffix_to_model_name,
        
        )
        
    def tune_models_and_get_results(models:dict,
                                    optimize:str,
                                    n_iter:int=3,
                                    results:Optional[pd.DataFrame]=None,
                                    suffix_to_model_name:Optional[str]=None, 
                                    )->Tuple[dict,pd.DataFrame]:
        """_summary_

        Args:
            models (dict): _description_
            optimize (str): _description_
            n_iter (int, optional): _description_. Defaults to 3.
            results (Optional[pd.DataFrame], optional): _description_. Defaults to None.
            suffix_to_model_name (Optional[str], optional): _description_. Defaults to None.

        Returns:
            Tuple[dict,pd.DataFrame]: _description_
        """        
        #https://pycaret.readthedocs.io/en/latest/api/classification.html#pycaret.classification.tune_model
        kwargs_model={'optimize':optimize,
                      'search_library':'scikit-optimize',
                      'n_iter':n_iter,
                      'choose_better':True,
                      'early_stopping':'asha',                   
                      }
        return train_the_model_with_specific_type_of_train_and_get_results(
            models=models,
            func="tune_model",
            results=results,
            suffix_to_model_name=suffix_to_model_name,
            **kwargs_model
        )
    
    def ensemble_models_and_get_results(models:dict,
                                    optimize:str,
                                    method:str, #Bagging or Boosting,
                                    results:Optional[pd.DataFrame]=None,
                                    suffix_to_model_name:Optional[str]=None,                                                                     
                                    )->Tuple[dict,pd.DataFrame]:
        """_summary_

        Args:
            models (dict): _description_
            optimize (str): _description_
            method (str): _description_
            results (Optional[pd.DataFrame], optional): _description_. Defaults to None.
            suffix_to_model_name (Optional[str], optional): _description_. Defaults to None.

        Returns:
            Tuple[dict,pd.DataFrame]: _description_
        """        
        #https://pycaret.readthedocs.io/en/latest/api/classification.html#pycaret.classification.ensemble_model
        kwargs_model={
            'optimize':optimize,
            'method':method ,#Bagging or Boosting,
            'choose_better':True,
                    }
        return train_the_model_with_specific_type_of_train_and_get_results(
            models=models,
            func="ensemble_model",
            results=results,
            suffix_to_model_name=suffix_to_model_name,
            **kwargs_model
        )
    
    def blend_models_and_get_results(models:dict,
                                    optimize:str,
                                    results:Optional[pd.DataFrame]=None,
                                    )->Tuple[dict,pd.DataFrame]:
        """_summary_

        Args:
            models (dict): _description_
            optimize (str): _description_
            method (str): _description_
            results (Optional[pd.DataFrame], optional): _description_. Defaults to None.

        Returns:
            Tuple[dict,pd.DataFrame]: _description_
        """        
        #https://pycaret.readthedocs.io/en/latest/api/classification.html#pycaret.classification.blend_model
        suffix_to_model_name='Blend'
 
        #models debe de ser el nombre y una lista
        models={ 'voting model':list(models.values())}
        kwargs_model={
            'optimize':optimize,
            'choose_better':False,
                    }
        models,results_per_cv= train_the_model_with_specific_type_of_train_and_get_results(
            models=models,
            func="blend_models",
            results=results,
            suffix_to_model_name=suffix_to_model_name,
            **kwargs_model
        )
        models={k:v for k,v in models.items() if not isinstance(v,list)}

        models,results_per_cv=tune_models_and_get_results(
            models,
            metric_to_use,
            n_iter=pycaret_atributtes['blend_parameters']['n_iter_in_tune'],
            results=results_per_cv,
            suffix_to_model_name="Tuned"
            )
     
        return models,results_per_cv
    
    def stack_models_and_get_results(models:dict,
                                    optimize:str,
                                    results:Optional[pd.DataFrame]=None,
                                    )->Tuple[dict,pd.DataFrame]:
        """_summary_

        Args:
            models (dict): _description_
            optimize (str): _description_
            method (str): _description_
            results (Optional[pd.DataFrame], optional): _description_. Defaults to None.

        Returns:
            Tuple[dict,pd.DataFrame]: _description_
        """        
        #https://pycaret.readthedocs.io/en/latest/api/classification.html#pycaret.classification.blend_model
        suffix_to_model_name='Stack'
 
        #models debe de ser el nombre y una lista
        models={ 'stack model':list(models.values())}
        meta_model=regression.create_model(pycaret_atributtes['stack_parameters']['meta_model'])
        kwargs_model={
            'optimize':optimize,
            'choose_better':True,
            'meta_model':meta_model 
                    }
        models,results_per_cv= train_the_model_with_specific_type_of_train_and_get_results(
            models=models,
            func="stack_models",
            results=results,
            suffix_to_model_name=suffix_to_model_name,
            **kwargs_model
        )
        models={f"{k}":v for k,v in models.items() if not isinstance(v,list)}
        models,results_per_cv=tune_models_and_get_results(
            models,
            metric_to_use,
            n_iter=pycaret_atributtes['stack_parameters']['n_iter_in_tune'],
            results=results_per_cv,
            suffix_to_model_name="Tuned"
            )
     
        return models,results_per_cv
    
    def create_normalized_df_with_cv_after_create_model (model_name:str)->pd.DataFrame:
        """_summary_

        Args:
            model_name (str): _description_

        Returns:
            pd.DataFrame: _description_
        """        
        df=regression.pull()
        df.index.name="fold"
        df.reset_index(inplace=True)
        
        model_name=model_name
        df["name_model"]=model_name 
        return df
    
    def get_model_name_from_model(model,suffix_to_model_name:Optional[str]='')->str:
        """_summary_

        Args:
            model (_type_): _description_
            suffix_to_model_name (Optional[str], optional): _description_. Defaults to ''.

        Returns:
            str: _description_
        """        
        model_name = type(model).__name__
        model_name=re.sub('([A-Z])', r' \1', model_name)
        
        model_name=add_suffix_to_model_name(model_name,model,suffix_to_model_name)
            # extra_info= str(model.sampling_strategy) if hasattr(model,'sampling_strategy') else ''
                
            # model_name=model_name+" "+ suffix_to_model_name+" "+extra_info
        return model_name.strip()
    
    def add_suffix_to_model_name(model_name:str,model,suffix_to_model_name:Optional[str]='')->str:
            
        if suffix_to_model_name:
            extra_info= str(model.sampling_strategy) if hasattr(model,'sampling_strategy') else ''
                
            model_name=model_name+" "+ suffix_to_model_name+" "+extra_info
        return model_name
    
    def get_best_n_models(results:pd.DataFrame,
                          models:dict,
                          metric:str,
                          number_models:int,
                          output_is_dict:bool=False,
                          all_models=None) ->Union[dict, list]:
        if results:
            result_aux=results[results["fold"]=="Mean"].copy()
        else:
            return all_models[:number_models]
        try:
            result_aux.sort_values(metric,inplace=True)
            logging.debug(result_aux.head(15))
        except Exception as e:
            logging.error(e)
            logging.debug(result_aux.head(15))
        name_models=result_aux.name_model[:number_models]
        logging.info(name_models)
        if output_is_dict:
            models_topn={}
            for name_model in name_models:
                models_topn[name_model]=models[name_model]
            return models_topn
        else:
            topn_models=[]
            for name_model in name_models:
                topn_models.append(models[name_model])
            return topn_models
        
    def fix_summary_df(results_per_cv:pd.DataFrame,metric_to_use:str)->pd.DataFrame:
        """_summary_

        Args:
            results_per_cv (pd.DataFrame): _description_
            metric_to_use (str): _description_

        Returns:
            pd.DataFrame: _description_
        """        
        summary_df_mean=results_per_cv[results_per_cv['fold']=='Mean' ].copy()
        summary_df_mean.pop('fold')
        summary_df_sd=results_per_cv[results_per_cv['fold']=='SD' ].copy()
        summary_df_sd.pop('fold')
        summary_df=pd.merge(summary_df_mean,summary_df_sd,on='name',suffixes=('_mean','_sd'))
        summary_df = summary_df.reindex(sorted(summary_df.columns), axis=1)
        #sorting columns
        print(summary_df.columns)
        first_column=summary_df.pop('name')
        summary_df.insert(0,'name',first_column)
        second_column=summary_df.pop(f'{metric_to_use}_mean')
        summary_df.insert(1,f'{metric_to_use}_mean',second_column)
        third_column=summary_df.pop(f'{metric_to_use}_sd')
        summary_df.insert(2,f'{metric_to_use}_sd',third_column)
        print(summary_df.head())

        return summary_df
    

    pycaret_atributtes=get_pycaret_attributes()
    setup_pycaret(df_train,pycaret_atributtes)
    logging.info("comparing models")
    metric_to_use=pycaret_atributtes['metric_to_use']
    all_models=regression.compare_models(
        exclude=pycaret_atributtes["models_exclude"],
        sort=metric_to_use ,
        n_select=pycaret_atributtes['n_models'],
        errors='raise'
        )
    
    compare_models_report=regression.pull()
    results_per_cv=None
    logging.info(compare_models_report.head(10))
    if not isinstance(all_models,list):
        all_models=[all_models]
    models= { get_model_name_from_model(model):model for model in all_models}
    # results_per_cv=pd.DataFrame()
    # _,results_per_cv=create_models_and_get_results(models)
    
    # logging.info(f' \n {results_per_cv.to_string()}')
    # print(results_per_cv.name_model.unique())
    if pycaret_atributtes['tune_parameters']['is_used']:
        logging.info("Tuning top n models")
        
        models,results_per_cv=tune_models_and_get_results(
            models,
            metric_to_use,
            n_iter=pycaret_atributtes['tune_parameters']['n_iter'],
            results=results_per_cv,
            suffix_to_model_name="Tuned"
            )
        logging.info(f' \n {results_per_cv.to_string()}')
        print(results_per_cv.name_model.unique())
    # topn_models=get_best_n_models(results_from_topn_models_pycaret_per_cv,models,metric_to_use,2)
    
    if pycaret_atributtes['ensemble_parameters']['is_used']:
        number_models_to_ensemble=pycaret_atributtes['ensemble_parameters']['number_topmodels']
        logging.info(f"ensembling top {number_models_to_ensemble} models")
        models_to_ensemble=get_best_n_models(results_per_cv,
                                              models,metric_to_use,
                                              number_models=number_models_to_ensemble,
                                              output_is_dict=True)
        
        for method in pycaret_atributtes['ensemble_parameters']["methods"]:
            new_models,results_per_cv=ensemble_models_and_get_results(
                models_to_ensemble,
                metric_to_use,
                method=method,
                results=results_per_cv,
                suffix_to_model_name=method
                )
            models={**models,**new_models}
        logging.info(f' \n {results_per_cv.to_string()}')
    
    if pycaret_atributtes['blend_parameters']['is_used']:
        number_models_to_blend=pycaret_atributtes['blend_parameters']['number_topmodels']
        logging.info(f"blending top {number_models_to_blend} models")
        models_to_blend=get_best_n_models(results_per_cv,
                                              models,metric_to_use,
                                              number_models=number_models_to_blend,
                                              output_is_dict=True)
        
        new_models,results_per_cv=blend_models_and_get_results(
            models_to_blend,
            metric_to_use,
            results=results_per_cv,
            )
        models={**models,**new_models}
        logging.info(f' \n {results_per_cv.to_string()}')
        
    if pycaret_atributtes['stack_parameters']['is_used']:
        number_models_to_blend=pycaret_atributtes['stack_parameters']['number_topmodels']
        logging.info(f"blending top {number_models_to_blend} models")
        models_to_blend=get_best_n_models(results_per_cv,
                                              models,metric_to_use,
                                              number_models=number_models_to_blend,
                                              output_is_dict=True)
        
        new_models,results_per_cv=stack_models_and_get_results(
            models_to_blend,
            metric_to_use,
            results=results_per_cv,
            )
        models={**models,**new_models}
        logging.info(f' \n {results_per_cv.to_string()}')
    # logging.info(scores.head())
    # blend top n base models 
    # logging.info("blending top n models")
    # blender = classification.     (estimator_list = topn) 
    # select best model 
    n=3
    logging.info(f"getting the best {n} model")
    topn_models=get_best_n_models(results_per_cv,models,metric_to_use,n,all_models=all_models)
    best_models = [regression.finalize_model(top_m) for top_m in topn_models]
    
    scores=regression.pull()
    # logging.info(models_report)
    logging.info(scores.head())
    try:
        first_column=results_per_cv.pop('name_model')
        results_per_cv.insert(0,'name',first_column)
        second_column=results_per_cv.pop(metric_to_use)
        results_per_cv.insert(1,metric_to_use,second_column)
        summary_df=fix_summary_df(results_per_cv,metric_to_use)

        return summary_df,results_per_cv,topn_models
    except:
        return scores,scores,topn_models

def check_model_with_pycaret(train,test:pd.DataFrame,model,text_to_identify_model:str,error_df,D_or_W):
    predictions=predict(model,train,start_date='2020-01-25',D_or_W=D_or_W)
    
    for id_unique in tqdm(predictions.ID.unique(),mininterval=25):
        predictions_per_id=predictions[predictions['ID']==id_unique]
        test_per_id=test[test['ID']==id_unique]
        error=rmse(test_per_id.total,predictions_per_id.Label)
        error_df.loc[id_unique,text_to_identify_model]=error
    error_df.index.name='ID'
    predictions['model_to_prediction']=text_to_identify_model
    return error_df,predictions
def run_pycaret_with_specific_distribution(df:pd.DataFrame,
    type_of_dataset='',
    D_or_W='D',
    )->Tuple[pd.DataFrame,pd.DataFrame,list]:
    def basic_run_pycaret(df,
        type_of_dataset,
        root_model=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models\best_model_',
        root_reporting=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting',
        D_or_W='D',
    ):
        df_train,df_test=split_data(df.copy(),column_date='date')
        if os.path.exists(root_model+type_of_dataset+'_'+str(0)+'.pkl'):
            models_paths=[]
            models_paths.append(root_model+type_of_dataset+'_'+str(0))
            models_paths.append(root_model+type_of_dataset+'_'+str(1))
            models_paths.append(root_model+type_of_dataset+'_'+str(2))
            models=[regression.load_model(path) for path in models_paths]
        else:

            
            outputs_train=run_pycaret(df_train,)
            df_summary=outputs_train[0]
            df_extensive=outputs_train[1]

            logging.info('saving results summary')
            
            summary='results_summary_'+type_of_dataset+"_"
            today= datetime.today().strftime('%Y-%m-%d')

            df_summary.to_csv(os.path.join(root_reporting,summary+today+'.csv'),index=False)
            df_extensive=outputs_train[1]
            logging.info('saving results_summary_per_fold')
            summary_extensive='results_summary_extensive_'+type_of_dataset+"_"
            today= datetime.today().strftime('%Y-%m-%d')
            df_extensive.to_csv(os.path.join(root_reporting,summary_extensive+today+'.csv'),index=False)
            models_list=outputs_train[2:]
            models=models_list[0]
            
            for i,model in enumerate(models):
                logging.info('saving the best model')
                path=root_model+type_of_dataset+'_'+str(i)
                regression.save_model(model,path)
        errors_total=pd.DataFrame()
        predictions_total=pd.DataFrame()
        for i,model in enumerate(models):
            path_folder_save_error_csv=os.path.join(root_reporting,'error',)
            os.makedirs(path_folder_save_error_csv,exist_ok=True)
            name_file_save_error_csv=f'pycaret_{type_of_dataset}_{i}.csv'
            pfile_save_error_csv=os.path.join(path_folder_save_error_csv,name_file_save_error_csv,)
            
            path_folder_prediction=os.path.join(root_reporting,'prediction',f'pycaret_{type_of_dataset}_{i}.csv')
            path_folder_save_prediction_csv=os.path.join(root_reporting,'prediction',)
            os.makedirs(path_folder_save_prediction_csv,exist_ok=True)
            name_file_save_prediction_csv=f'pycaret_{type_of_dataset}_{i}.csv'
            pfile_save_prediction_csv=os.path.join(path_folder_save_prediction_csv,name_file_save_prediction_csv,)

            if os.path.exists(pfile_save_error_csv) and os.path.exists(pfile_save_prediction_csv):
                errors_total=pd.read_csv(pfile_save_error_csv)
                predictions=pd.read_csv(pfile_save_prediction_csv)
            else:
                errors_total,predictions=check_model_with_pycaret(df_train,
                    df_test,model,
                    f'pycaret_{type_of_dataset}_{i}',
                    errors_total.copy(),
                    D_or_W)
                # errors_total=pd.concat([errors_total,errors])
                errors_total.to_csv(pfile_save_error_csv)
                predictions.to_csv(pfile_save_prediction_csv)
        predictions_total=pd.concat([predictions_total,predictions])
            
        return errors_total,predictions_total

    path_pycaret_error=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting\errors\errors_'+type_of_dataset+'.csv'
    path_pycaret_error=modify_path_if_week_periocity(path_pycaret_error, D_or_W)
    path_pycaret_predictions=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\07_model_outputs\predictions_'+type_of_dataset+'.csv'
    path_pycaret_predictions=modify_path_if_week_periocity(path_pycaret_predictions, D_or_W)
    if os.path.isfile(path_pycaret_error) and os.path.isfile(path_pycaret_predictions):
        errors_total=pd.read_csv(path_pycaret_error)
        predictions_total=pd.read_csv(path_pycaret_predictions)
    else:
        if type_of_dataset=='all_data':
            # df=df[df['ID']<10]
            
            root_model=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models\all_data'
            root_model=modify_path_if_week_periocity(root_model, D_or_W)
            prefix='best_model_'
            os.makedirs(root_model,exist_ok=True)
            root_model=os.path.join(root_model,prefix)
            root_reporting=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting\all_data'
            root_reporting=modify_path_if_week_periocity(root_reporting, D_or_W)
            os.makedirs(root_reporting,exist_ok=True)
            
            errors_total,predictions_total=basic_run_pycaret(
                df,
                type_of_dataset,
                root_model,
                root_reporting,
                D_or_W
                )
            errors_total.to_csv(path_pycaret_error)
            predictions_total.to_csv(path_pycaret_predictions,index=False)
        elif type_of_dataset=='per_clusters_mean':
            errors_total=pd.DataFrame()
            predictions_total=pd.DataFrame()
            root_model=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models\cluster_mean_group'
            root_model=modify_path_if_week_periocity(root_model, D_or_W)
            prefix='best_model_'
            os.makedirs(root_model,exist_ok=True)
            root_model=os.path.join(root_model,prefix)
            
            
            root_reporting=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting\cluster_mean_group'
            root_reporting=modify_path_if_week_periocity(root_reporting, D_or_W)
            os.makedirs(root_reporting,exist_ok=True)
            for mean in tqdm(df.group_by_mean_in_total.unique(),desc='training model per mean'):
                df_aux=df[df['group_by_mean_in_total']==mean]
                type_of_dataset_to_this_group=type_of_dataset+f'_group_{mean}'
                try:
                    errors_df_per_group,predictions_df_per_group=basic_run_pycaret(
                        df_aux,
                        type_of_dataset_to_this_group,
                        root_model,
                        root_reporting,
                        D_or_W,
                        )
                    print(errors_df_per_group.head())
                except Exception as e:
                    print(e)
                    print(df.group_by_mean_in_total.unique())
                    print(df_aux.head(5))
                errors_total=pd.concat([errors_total,errors_df_per_group])
                print(errors_total.head())
                predictions_total=pd.concat([predictions_total,predictions_df_per_group])

            errors_total.to_csv(path_pycaret_error,index=False)
            predictions_total.to_csv(path_pycaret_predictions,index=False)

        elif type_of_dataset=='per_clusters_std':
            
            errors_total=pd.DataFrame()
            predictions_total=pd.DataFrame()
 
            root_model=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models\cluster_std_group'
            root_model=modify_path_if_week_periocity(root_model, D_or_W)
            prefix='best_model_'
            os.makedirs(root_model,exist_ok=True)
            root_model=os.path.join(root_model,prefix)
            root_reporting=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting\cluster_std_group'
            root_reporting=modify_path_if_week_periocity(root_reporting, D_or_W)
            os.makedirs(root_reporting,exist_ok=True)
            for std in tqdm(df.group_by_std_in_total.unique(),desc='training model per std'):
                df_aux=df[df['group_by_std_in_total']==std]
                type_of_dataset_to_this_group=type_of_dataset+f'_group_{std}'
                errors_df_per_group,predictions_df_per_group=basic_run_pycaret(df_aux,
                    type_of_dataset_to_this_group,
                    root_model,
                    root_reporting,
                    D_or_W,
                    )
                if 'ID.1' in errors_df_per_group.columns:
                    print('error')
                errors_total=pd.concat([errors_total,errors_df_per_group])
                if 'ID.1' in errors_total.columns:
                    print('error')
                predictions_total=pd.concat([predictions_total,predictions_df_per_group])
            
            errors_total.to_csv(path_pycaret_error,index=False)
            predictions_total.to_csv(path_pycaret_predictions,index=False)

        else:
            raise 'option no valid'
    return errors_total,predictions_total

def retrain_model(df,path_model,type_model,id_unique,D_or_W='D',):
    root_model=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models'
    
    folder_the_definitive_model='last_model_per_id'
    path_to_save_the_model_to_prediction=os.path.join(root_model,folder_the_definitive_model)
    path_to_save_the_model_to_prediction=modify_path_if_week_periocity(path_to_save_the_model_to_prediction, D_or_W)
    os.makedirs(path_to_save_the_model_to_prediction,exist_ok=True)
    name_of_model=os.path.split(path_model)[-1]
    path_model_last_version=os.path.join(path_to_save_the_model_to_prediction,name_of_model)
    
    if type_model=='pycaret_regression' :
        if not os.path.isfile(path_model_last_version+'.pkl'):
            shutil.copy(path_model+'.pkl',path_model_last_version+'.pkl')
        # pycaret_atributtes=get_pycaret_attributes()
        # model=regression.load_model(path_model)
        #crear el setup, luego create model y luego el finalize y guardar el modelo

        # df_train,df_test=split_data(df.copy(),column_date='date')
        # predictions=regression.predict_model(model,df_test)
        # error=rmse(df_test.total,predictions.Label)

        # new_model=model.fit(df,df.total)
        # # new_model[-1][1].
        # predictions=regression.predict_model(new_model,df_test)
        # error=rmse(df_test.total,predictions.Label)
        # setup_pycaret(df,pycaret_atributtes)
        # new_model=regression.finalize_model(model)
        
        # # a=df_test.drop('total',axis=1)
        
        
        
        # regression.save_model(model,path_model_last_version)
        # print(model)
        
    elif type_model=='prophet':
        if not os.path.isfile(path_model_last_version):
            df_aux=df[df['ID']==id_unique]
            pivot_by_dates=transform_df_in_pivot_with_id_in_columns(df_aux)
            df_prophet=pd.DataFrame()
            df_prophet[['ds','y']]=pivot_by_dates.reset_index()
            if D_or_W=='D':
                weekly_seasonality=True
            else:
                weekly_seasonality=Fals
            if os.path.exists(path_model):
                try:
                    with open(path_model, 'r') as fin:
                        m_prev = model_from_json(json.load(fin))  # Load model   
                    m = Prophet(yearly_seasonality=False,
                            weekly_seasonality=True,
                            daily_seasonality=False,
                            )
                
                    m.fit(df_prophet,init=stan_init(m_prev))
                    with open(path_model_last_version, 'w') as fout:
                        json.dump(model_to_json(m), fout)  # Save model
                except Exception as e:
                    logging.error('una excepción donde prophet')
                    logging.error(e)
                    m = Prophet(yearly_seasonality=False,
                    weekly_seasonality=True,
                    daily_seasonality=False
                    )
                    m.fit(df_prophet)
                    with open(path_model_last_version, 'w') as fout:
                        json.dump(model_to_json(m), fout)  # Save model
            else:
                raise 'En teoría ya debe de estar entrenado'
    elif type_model=='clusters_mean' or type_model=='clusters_std':
        if not os.path.isfile(path_model_last_version+'.pkl'):
            shutil.copy(path_model+'.pkl',path_model_last_version+'.pkl')
        # model=regression.load_model(path_model)
        # regression.save_model(model,path_model_last_version)
