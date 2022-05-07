#https://tsfresh.readthedocs.io/en/latest/ ver esta librería
from cmath import exp
from copy import error
from http.client import OK
import logging
import multiprocessing
import os
import re
import time
from datetime import datetime
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from pycaret import regression
from sqlalchemy import desc
from tqdm import tqdm

from src.analysis import check_pca_in_df
from src.system_to_train import (run_pycaret_with_specific_distribution,
                                 train_systems_with_prophet_without_exogenous,
                                 train_systems_with_pycaret_regression,retrain_model)
from src.utils import applyParallel_using_df_standar, get_range_of_dates,modify_path_if_week_periocity

#https://towardsdatascience.com/time-series-feature-extraction-on-really-large-data-samples-b732f805ba0e
try:
    from tsfresh import extract_features
except:
    pass
logger=logging.getLogger(__name__)

def filter_ids_in_df_with_condition(df,get_id_less_than:Optional=None):
    list_of_id_with_tiny_data=[2521, 2724, 2725, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2739, 2743, 2744, 2746, 2747, 2749, 2756]
    if get_id_less_than:
        logging.info(f'getting df with equal and less than {get_id_less_than} ')
        df=df[df['ID']<=get_id_less_than]
    else:
        logging.info('removing list of id with fews data')
        df=df[~df['ID'].isin(list_of_id_with_tiny_data)]
    return df



def train_systems(df,D_or_W):
    def train_and_check_error_to_select_which_is_the_best_model_per_id(df,D_or_W):
        
        root_all_errors=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting\summary_all_errors.csv'
        root_all_errors=modify_path_if_week_periocity(root_all_errors, D_or_W)
        root_all_predictions=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting\summary_all_predictions.csv'
        root_all_predictions=modify_path_if_week_periocity(root_all_predictions, D_or_W)
        if os.path.exists(root_all_errors) and os.path.exists(root_all_predictions) and False:
            all_errors=pd.read_csv(root_all_errors)
            try:
                all_predictions=pd.read_csv(root_all_predictions)
            except:
                all_predictions=pd.DataFrame()
        else:
            
            errors=[]
            predictions=[]          
            
            error_df,predictions_df=run_pycaret_with_specific_distribution(df,type_of_dataset='all_data',D_or_W=D_or_W)
            errors.append(error_df)
            print(error_df.ID.nunique())
            predictions.append(predictions_df)
            try:
                error_df,predictions_df=run_pycaret_with_specific_distribution(df,type_of_dataset='per_clusters_mean',D_or_W=D_or_W)
                print(error_df.head())
                print(error_df.ID.nunique())
                errors.append(error_df)
                predictions.append(predictions_df)
            except Exception as e:
                print(e)
                print('error encluster mean')
            try:

            
                error_df,predictions_df=run_pycaret_with_specific_distribution(df,type_of_dataset='per_clusters_std',D_or_W=D_or_W)
                errors.append(error_df)
                predictions.append(predictions_df)
                print(error_df.ID.nunique())
            except Exception as e:
                print(e)
                print('rror en clustes std')
            try:
                if D_or_W=='D':
                    error_df,predictions_df=train_systems_with_prophet_without_exogenous(df,D_or_W=D_or_W)
                    errors.append(error_df)
                    predictions.append(predictions_df)
                    print(error_df.ID.nunique())
            except Exception as e:
                print(e)
                print('rror en prophet')
            
            error_df,predictions_df=train_systems_with_pycaret_regression(df,D_or_W='D') #need to much  time
            errors.append(error_df)
            predictions.append(predictions_df)
            
            all_errors=pd.DataFrame()
            for df in errors:
                if all_errors.empty:
                    all_errors=df
                else:
                    all_errors=pd.merge(all_errors,df,on='ID')
            print(all_errors.ID.nunique())        
            print(all_errors.head())
            all_errors['min_error']=all_errors.drop('ID',axis=1).min(axis=1)
            all_errors['best_model']=all_errors.drop('ID',axis=1).idxmin(axis=1)
            logging.info(all_errors.sum())
            all_errors.to_csv(root_all_errors,index=False)
            
            all_predictions=pd.DataFrame()
            for df in all_predictions:
                if all_predictions.empty:
                    all_predictions=df
                else:
                    all_predictions=pd.merge(all_predictions,df,on='ID')
            all_predictions.to_csv(root_all_predictions,index=False)

    def retrain(df,D_or_W):

        root_all_errors=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting\summary_all_errors.csv'
        root_all_errors=modify_path_if_week_periocity(root_all_errors, D_or_W)
        root_models=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models'
        # root_models=modify_path_if_week_periocity(root_models, D_or_W)
        if os.path.exists(root_all_errors):
            df_summary_errors=pd.read_csv(root_all_errors)

            for id_unique in tqdm(df_summary_errors.ID.unique(),mininterval=25,desc="retrain_models"):
                try:
                    df_summary_errors_aux=df_summary_errors[df_summary_errors['ID']==id_unique]
                    model_choose=df_summary_errors_aux['best_model'].values[0]
                    print(model_choose)

                    if  'pycaret_all_data' in model_choose:
                        extension_model=''
                        folder_model='all_data'
                        which_model=str(model_choose.split("_")[-1])
                        
                        name_model=f'best_model_all_data_{which_model}{extension_model}'
                        path_model=os.path.join(root_models,folder_model)
                        path_to_model=modify_path_if_week_periocity(path_model, D_or_W)
                        path_to_this_model=os.path.join(path_to_model,name_model)
                        type_model='pycaret_regression'

                    elif  'Prophet' in model_choose :
                        extension_model='.json'
                        folder_model='prophet'
                        name_model=f'train_without_exogenous_variables_serialized_model_id_{id_unique}{extension_model}'
                        path_model=os.path.join(root_models,folder_model)
                        path_to_model=modify_path_if_week_periocity(path_model, D_or_W)
                        path_to_this_model=os.path.join(path_to_model,name_model)
                        type_model='prophet'

                    elif  'clusters_mean' in model_choose :
                        extension_model=''
                        folder_model='cluster_mean_group'
                        which_model=model_choose.split("_")[-2:]
                        which_model='_'.join(which_model)
                        name_model=f'best_model_per_clusters_mean_group_{which_model}{extension_model}'
                        path_model=os.path.join(root_models,folder_model)
                        path_to_model=modify_path_if_week_periocity(path_model, D_or_W)
                        path_to_this_model=os.path.join(path_to_model,name_model)
                        type_model='clusters_mean'
                    elif  'clusters_std' in model_choose :
                        extension_model=''
                        folder_model='cluster_std_group'
                        which_model_raw=model_choose.split("_")
                        which_model='_'.join(which_model_raw[-2:])
                        name_model=f'best_model_per_clusters_std_group_{which_model}{extension_model}'
                        path_model=os.path.join(root_models,folder_model)
                        path_to_model=modify_path_if_week_periocity(path_model, D_or_W)
                        path_to_this_model=os.path.join(path_to_model,name_model)
                        type_model='clusters_std'
                    retrain_model(df,path_to_this_model,type_model,id_unique,D_or_W)
                except Exception as e:
                    print(id_unique)
                    logging.error(e)
        else:
            raise 'we need the summary'
        return df_summary_errors
    #comprobar el error para los mejores modelos por cada ID
    start = time.time()

    train_and_check_error_to_select_which_is_the_best_model_per_id(df,D_or_W)
    df_summary_errors=retrain(df,D_or_W)
    end = time.time()
    logging.info(f'time to train model {(end - start)/60} min')
    
    return df_summary_errors

