import time
from importlib.resources import path

from dotenv import load_dotenv

load_dotenv(r'D:\programacion\Repositorios\datathon-cajamar-2022\conf\local\.env')
import os
import os.path
# print(os.environ)
import sys

sys.path.append(r'D:\programacion\Repositorios\datathon-cajamar-2022')
import logging
import warnings

import pandas as pd
from pycaret import regression

from src.create_the_output_file import create_outputfile
from src.generate_df import generate_df
from src.pipeline import  train_systems
from src.predict import predict_per_id
from src.utils import extract_n_relevant_features,modify_path_if_week_periocity
from src.create_features import create_features

warnings.simplefilter(action='ignore', category=FutureWarning)

logging.getLogger().setLevel(logging.DEBUG)
def main(D_or_W='D'):
    try:
        logging.info('creating features')
        start = time.time()
        path_df_primary=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\03_primary\to_train.csv'
        path_df_primary=modify_path_if_week_periocity(path_df_primary,D_or_W)
        if os.path.isfile(path_df_primary):
            logging.info('loading features')
            df_with_features=pd.read_csv(path_df_primary)
            df_with_features.loc[:,'date']=pd.to_datetime(df_with_features['date'],errors='raise',format='%Y-%m-%d')
            logging.info(df_with_features.ID.nunique())
            # df_with_features=filter_ids_in_df_with_condition(df_with_features,get_id_less_than=5)
            logging.info(df_with_features.ID.nunique())
            logging.info(df_with_features.shape)
            logging.info(df_with_features.columns.to_list())
        else:
            df=generate_df(D_or_W=D_or_W)
            df_with_features=create_features(df,D_or_W=D_or_W)
            logging.info('saving primary/features')
            df_with_features.to_csv(path_df_primary,index=False)
        logging.info('getting attributes to insert in pycaret')
        end = time.time()
        logging.info(f'time to create features {(end - start)/60} min')
        start = time.time()
        #analysis
        # pycaret_attributes=get_pycaret_attributes()

        train_systems(df_with_features,D_or_W)
        

        path_prediction=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\07_model_outputs\predictions.csv'
        path_prediction=modify_path_if_week_periocity(path_prediction,D_or_W)
        if os.path.isfile(path_prediction) and False:
            predictions=pd.read_csv(path_prediction)
            print(predictions.ID.nunique())
            logging.info(predictions.columns.to_list())
        else:
            root_all_errors=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting\summary_all_errors.csv'
            root_all_errors=modify_path_if_week_periocity(root_all_errors, D_or_W)
            if os.path.exists(root_all_errors):
                df_summary_errors=pd.read_csv(root_all_errors)
            predictions=predict_per_id(df_with_features,df_summary_errors,D_or_W)
            predictions.to_csv(path_prediction,index=False)
        
        
        
        print('hi')
    except Exception as e:
        print(e)
    

def creating_outputfile_with_daily_and_weekly_predictions():
    D_or_W='D'
    path_prediction=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\07_model_outputs\days\predictions.csv'
    # path_prediction=modify_path_if_week_periocity(path_prediction,D_or_W)
    predictions_daily=pd.read_csv(path_prediction)
    predictions_daily.loc[:,'date']=pd.to_datetime(predictions_daily['date'],errors='raise',format='%Y-%m-%d')
    print(predictions_daily.ID.nunique())
    print(predictions_daily.date.nunique())
    logging.info(predictions_daily.columns.to_list())
    D_or_W='W'
    path_prediction=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\07_model_outputs\week\predictions.csv'
    # path_prediction=modify_path_if_week_periocity(path_prediction,D_or_W)
    predictions_weekly=pd.read_csv(path_prediction)
    print(predictions_weekly.ID.nunique())
    print(predictions_daily.date.nunique())
    # logging.info(predictions_weekly.columns.to_list())
    create_outputfile(predictions_daily,predictions_weekly,path_output='CanarIAs.txt')
 


if __name__=='__main__':
    main('D')
    main('W')
    creating_outputfile_with_daily_and_weekly_predictions()
    
