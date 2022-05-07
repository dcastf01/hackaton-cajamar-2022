
import json
import logging
import multiprocessing
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.utils import (applyParallel_using_df_standar, load_model, save_model,modify_path_if_week_periocity,
                       stan_init)

warnings.filterwarnings("ignore")
def create_features(df_original,target='total',is_to_predict:bool=False,drop_dates=None,D_or_W='D'):
    
    """
    Creates time series features from datetime index
    """
    def select_features(df):
        logging.info('selecting features')
        data_without_target_variable=df.copy()
        data_without_target_variable.drop(target,axis=1)

        corr_matrix = data_without_target_variable.corr()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        logging.info(f'removing { len(to_drop)}')
        logging.info(f'shape before drop {df.shape}')
        df_after_drop=df.drop(to_drop, axis=1)
        logging.info(f'shape after drop {df.shape}')
        return df_after_drop

    logging.info('creating features')
    df=df_original.copy()
    df.loc[:,'date']=pd.to_datetime(df['date'],errors='raise',format='%Y-%m-%d')
    df=basic_temporal_features(df)
    print(df.head())

    df_with_group_by_mean_in_total=cluster_by_stadistic_variable(df,'mean',D_or_W=D_or_W)
    df_with_group_by_std_in_total=cluster_by_stadistic_variable(df,'std',D_or_W=D_or_W)

    features_manually_temporal_variables=extra_features_manually_temporal_variables(
        df,
        is_to_predict=is_to_predict,
        D_or_W=D_or_W
        )
    features_manually_temporal_variables=apply_pca_in_manual_temporal_features(features_manually_temporal_variables,
        components_pca=5,
        is_to_predict=is_to_predict,
        D_or_W=D_or_W
        )
    features_manually_temporal_variables.loc[:,'date']=pd.to_datetime(
        features_manually_temporal_variables['date'],errors='raise',format='%Y-%m-%d')
    # check_pca_in_df(features_manually_temporal_variables.drop(['ID','date'],axis=1),'temporal_variables')
    features_no_temporal=extract_features_no_temporal(df,D_or_W=D_or_W)
    
    # features_using_prophet=extract_features_from_prophet(df,is_to_predict=is_to_predict)

    columns_to_keep=selecting_columns_to_drop()
    
    #añadir la suma del mes anterior
    #añadir la suma de la semana anterior
    #añadir la suma del trimestre anterior
    logging.info(f'shape of features features_no_temporal BEFORE clean using feature selection {features_no_temporal.shape}')
    # for col in features_tsfresh_per_id.columns:
    #     if col not in  columns_to_keep and col in features_tsfresh_per_id.columns:
    #         features_tsfresh_per_id.drop(col,axis=1,inplace=True)
    logging.info(f'shape of features features_no_temporal AFTER clean using feature selection {features_no_temporal.shape}')
    logging.info(f'shape of features features_manually_temporal_variables BEFORE clean using feature selection {features_manually_temporal_variables.shape}')
    # for col in features_manually_temporal_variables.columns:
    #     if col not  in  columns_to_keep and col in features_manually_temporal_variables.columns:
    #         features_manually_temporal_variables.drop(col,axis=1,inplace=True)
    logging.info(f'shape of features features_manually_temporal_variables AFTER clean using feature selection {features_manually_temporal_variables.shape}')
    #add data from tsfresh y from prophet (en ambos casos se hará por id y luego se mergeará a ese ID)
    X = df#[[
        # 'dayofweek',
        # 'quarter',
        # 'month',
            # 'year',  #only we have a year, and that is not enought to the model learn what hapen to the next year
        # 'dayofyear',
        # 'dayofmonth',
        # 'weekofyear',
        # 'Monday',
        # 'Friday','Saturday','Sunday','Thursday','Tuesday','Wednesday',
        # target,'ID','date',
        # ]]
    logging.info('merging X with tsfresh')
    print(X.ID.nunique())
    # X=pd.merge(X,features_using_prophet,left_on=['ID','date'],right_on=['ID','date'],how='left')
    print(X.head())
    logging.info('merging X with features_manually_no_temporal')
    print(X.ID.nunique())
    X=pd.merge(X,features_no_temporal,right_on='ID',left_on='ID',how='left')
    X=pd.merge(X,df_with_group_by_mean_in_total,right_on='ID',left_on='ID',how='left')
    X=pd.merge(X,df_with_group_by_std_in_total,right_on='ID',left_on='ID',how='left')

    print(X.head())
    logging.info('merging X with features_manually_temporal_variables')
    print(features_manually_temporal_variables.head())
   
    X=pd.merge(X,features_manually_temporal_variables,right_on=['ID','date'],left_on=['ID','date'],how='left')
    
    if drop_dates:
        X=X[X['date']<drop_dates]
    X.sort_values(['ID','date'],inplace=True)
    X.reset_index(inplace=True)
    print(X.columns)
    print(X.head())
    X.drop('index',axis=1,inplace=True)
    if 'index_y' in X.columns:
        X.drop('index_y',axis=1,inplace=True) #checkear de donde viene, casi seguro que de df_with_group
    if 'index_x' in X.columns:
        X.drop('index_x',axis=1,inplace=True)
    print(X.columns)
    # X=select_features(X)

    return X

def cluster_by_stadistic_variable(df,kind,D_or_W):
    #kind std or mean
    path_csv_cluster_by_stadistic_variable=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\04_feature\ID_per_stadistic_variable'
    path_csv_cluster_by_stadistic_variable=path_csv_cluster_by_stadistic_variable+f'_{kind}.csv'
    path_csv_cluster_by_stadistic_variable=modify_path_if_week_periocity(path_csv_cluster_by_stadistic_variable,D_or_W)
    if os.path.isfile(path_csv_cluster_by_stadistic_variable):
        df_final=pd.read_csv(path_csv_cluster_by_stadistic_variable)

        return df_final
    else:
        if D_or_W=='W':
            max_clusters=25 #we have less data
        else:
            max_clusters=50
        df_total_to_pivot=df.reset_index()
        pivot_by_dates=df_total_to_pivot.pivot(
        index='date',
        columns='ID',
        values='total'
        )
        describe_by_dates=pivot_by_dates.describe()
        group_2000=describe_by_dates.T[describe_by_dates.T[kind]>=2000]
        group_1000_2000=describe_by_dates.T[(describe_by_dates.T[kind]>1000)&(describe_by_dates.T[kind]<2000)]
        group_0=describe_by_dates.T[describe_by_dates.T[kind]==0]
        rest_df=describe_by_dates.T[(describe_by_dates.T[kind]<=1000)&(describe_by_dates.T[kind]>0)]
        group_0[f'group_by_{kind}_in_total']=1
        rest_df[f'group_by_{kind}_in_total']=pd.qcut(rest_df[kind],q=max_clusters,labels=range(2,max_clusters+2))
        logging.info(rest_df.isna().sum())
        group_1000_2000[f'group_by_{kind}_in_total']=max_clusters+2
        group_2000[f'group_by_{kind}_in_total']=max_clusters+3
        df_final=pd.concat([group_0,rest_df,group_1000_2000,group_2000])[f'group_by_{kind}_in_total']
        df_final.index.name='ID'
        # print(df_final.ID.nunique())
        df_final.to_csv(path_csv_cluster_by_stadistic_variable,index=True)
        return df_final.reset_index()

# def cheking_outliers(df):
#     #Deprecated
#     path_csv_after_outliers=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\04_feature\df_after_outliers.csv'
#     if os.path.isfile(path_csv_after_outliers):
#         df_aux=pd.read_csv(path_csv_after_outliers)
#     else:
#         from sklearn.ensemble import IsolationForest

#         outliers_fraction=0.1
#         method_forest=IsolationForest(contamination=outliers_fraction)
#         df_to_use=df[['total','ID','date']]
#         df_aux=pd.DataFrame()
#         def checking_outliers(df,id_unique):
    
#             df_per_id=df_to_use[df_to_use['ID']==id_unique].copy()
#             add_later=df_per_id['date']
#             df_per_id.drop('date',axis=1,inplace=True)
#             data_to_correction=df_per_id['total'].rolling(window=14,closed='left').mean().fillna(method='bfill')
#             yhat = method_forest.fit_predict(df_per_id)
#             mask = yhat != -1
#             df_per_id['total']=np.where(mask,
#                                         df_per_id.total,
#                                         data_to_correction
#                                     )
#             df_per_id.loc[:,'date']=add_later
#             return df_per_id
#             # df_aux=pd.concat([df_aux,df_per_id],axis=0)
#         df_aux=applyParallel_using_df_standar(df,checking_outliers,njobs=10,desc='cheking outliers')
#         df_aux.to_csv(path_csv_after_outliers,index=False)

#     return df_aux.reset_index()
def basic_temporal_features(df):
    # dfdf['year']=df['SAMPLETIME'].dt.year
    logging.info('creating basic temporal features')
    # df['month']=df['date'].dt.month
    # df['day']=df['date'].dt.day
    # df['dayofweek'] = df['date'].dt.dayofweek
    # df['quarter'] = df['date'].dt.quarter
    # df['month'] = df['date'].dt.month
    # df['year'] = df['date'].dt.year
    # df['dayofyear'] = df['date'].dt.dayofyear
    # df['dayofmonth'] = df['date'].dt.day
    # df['weekofyear'] = df['date'].dt.weekofyear
    # df=pd.concat((df, pd.get_dummies(df['date'].dt.day_name())), axis=1)

    return df
    
def extract_features_no_temporal(df,D_or_W):
    path_features_no_temporal=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\04_feature\pca_features_no_temporal.csv'
    path_features_no_temporal=modify_path_if_week_periocity(path_features_no_temporal,D_or_W)
    if os.path.isfile(path_features_no_temporal):
        X=pd.read_csv(path_features_no_temporal)
    else:
        features_tsfresh_per_id=extract_features_per_id_using_tsfresh(df)
        features_manually_no_temporal=extra_features_manually_no_temporal_variables(df,D_or_W=D_or_W)
        features_no_temporal=pd.merge(features_manually_no_temporal,features_tsfresh_per_id,left_on='ID',right_on='ID',how='left')
        path_save_model= r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models\pca_features_no_temporal.pkl'
        path_save_model=modify_path_if_week_periocity(path_save_model,D_or_W)
        pca_features_no_temporal=create_pca_components_without_date(features_no_temporal,path_save_model,prefix='no_temporal')
        X=pca_features_no_temporal
        # X=create_clusters_without_date(pca_features_no_temporal)
        X.to_csv(path_features_no_temporal,index=False)
    print(X.head())
    return X
def scaled_features(X,path_scaled_model=None,):
    if os.path.isfile(path_scaled_model):
        scaled_loaded=load_model(path_scaled_model)
        X_scaled=scaled_loaded.transform(X)
    else:
        scaled=StandardScaler()
        X_scaled=scaled.fit_transform(X)
        save_model(scaled,path_scaled_model)
    return X_scaled
def create_scaled_model_name(path):
    parts=path.split('.')
    parts[0]+='_scaled'
    path='.'.join(parts)
    return path
def create_pca_components_without_date(df,path_model:Optional[str]=None,n_components=2,prefix:str=''):
    to_add_later=df['ID']
    X=df.copy().drop(['ID'],axis=1)
    if os.path.isfile(path_model):
        path_scaled=create_scaled_model_name(path_model)
        X_scaled=scaled_features(X,path_scaled_model=path_scaled)
        model_loaded=load_model(path_model)
        X_ipca=model_loaded.transform(X_scaled)
        n_components=model_loaded.n_components
    else:
        path_scaled=create_scaled_model_name(path_model)
        X_scaled=scaled_features(X,path_scaled)
        ipca = IncrementalPCA(n_components=n_components, batch_size=100)
        X_ipca = ipca.fit_transform(X_scaled)
        save_model(ipca,path_model)
    columns=[f'pca_{prefix}_{i}'for i in range(1,n_components+1)]
    pca_df=pd.DataFrame(X_ipca,columns=columns)
    
    pca_df=pd.concat([to_add_later,pca_df],axis=1)
    return pca_df

def create_clusters_without_date(pca_df,n_cluster=18):
    to_add_later=pca_df
    X=pca_df.copy().drop(['ID'],axis=1)
    model = KMeans(n_clusters=n_cluster)
    yhat = model.fit_predict(X)
    X['cluster_no_temporal']=yhat.tolist()
    # pca_df=pd.concat([to_add_later,X],axis=1)
    # df=pd.concat([to_add_later,pd.get_dummies(X['cluster_no_temporal'])],axis=1)

    return df

def extract_features_per_id_using_tsfresh(df,target='total'):
    #'This step we need use in colab because we have a problems with gpu'
    logging.info('creating features per id using tsfresh')
    path_features_tsfresh=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\04_feature\features_tsfresh_per_id.csv'
    if os.path.isfile(path_features_tsfresh):
        X=pd.read_csv(path_features_tsfresh)
    else:
        from tsfresh import (extract_features, extract_relevant_features,
                             select_features)
        from tsfresh.feature_extraction import ComprehensiveFCParameters
        from tsfresh.utilities.dataframe_functions import impute
        df=df[['ID',target,'date']]
        # We are very explicit here and specify the `default_fc_parameters`. If you remove this argument,
        # the ComprehensiveFCParameters (= all feature calculators) will also be used as default.
        # Have a look into the documentation (https://tsfresh.readthedocs.io/en/latest/text/feature_extraction_settings.html)
        # or one of the other notebooks to learn more about this.
        extraction_settings = ComprehensiveFCParameters()

        X = extract_features(df, column_id='ID', column_sort='date',
                            default_fc_parameters=extraction_settings,
                            # we impute = remove all NaN features automatically
                            impute_function=impute)
        X.reset_index(inplace=True)
        X.rename({'index':'ID'},inplace=True,axis=1)
        X.to_csv(path_features_tsfresh,index=False)
    return X
# def extract_features_temporal(df,target='total'):
    
#     decomposition.seasonal.plot()
#     return df
def extract_features_from_prophet(df,target='total',is_to_predict:bool=False,D_or_W='D'):
    
        
    def process_to_extract_features_prophet(df_unique_id,id):
        
        # df_per_id=df[df['ID']==id]
        # df_unique_id
        df_unique_id.drop('ID',axis=1,inplace=True)
        root_path_model=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models\prophet'
        root_path_model=modify_path_if_week_periocity(root_path_model,D_or_W)
        fpath_model=f'{root_path_model}\to_extract_features_serialized_model_{id}.json'
 
        # print(df_unique_id)
        if os.path.exists(fpath_model):
            with open(fpath_model, 'r') as fin:
                m_prev = model_from_json(json.load(fin))  # Load model
            m = Prophet(yearly_seasonality=False,
                        weekly_seasonality=True,
                        daily_seasonality=False
                        )
           
            m.fit(df_unique_id,init=stan_init(m_prev))
        else:

            m = Prophet(yearly_seasonality=False,
                        weekly_seasonality=True,
                        daily_seasonality=False
                        )
           
            m.fit(df_unique_id)
        forecast = m.predict()
        print(forecast.head())
        print(forecast.columns)
        forecast["ID"]=id
        
        with open(fpath_model, 'w') as fout:
            json.dump(model_to_json(m), fout)  # Save model
        return forecast
    
    # def applyParallel_using_df_standar(df, func,njobs=-1):
    #     if njobs==-1 :n_jobs=multiprocessing.cpu_count()  
    #     retLst = Parallel(n_jobs=njobs)(delayed(func)(df[df['ID']==id],id) for id in tqdm(df.ID.unique(),
    #                                                                                     desc='extracting features from prophet per id',
    #                                                                                     mininterval=25))
    #     return pd.concat(retLst)
    #https://stackoverflow.com/questions/48875601/group-by-apply-with-pandas-and-multiprocessing
    logging.info('creating features using prophet')
    path_features_prophet=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\02_intermediate\features_from_prophet.csv'
    path_features_prophet=modify_path_if_week_periocity(path_features_prophet,D_or_W)
    if os.path.isfile(path_features_prophet) and not is_to_predict:
        df_aux=pd.read_csv(path_features_prophet)
        df_aux['ds']=pd.to_datetime(df_aux['ds'],format='%Y-%m-%d')
    else:
        df_aux=pd.DataFrame()
        # df
        
        count_row_per_id=df.groupby('ID').count().reset_index()
        id_less_than_3_rows=count_row_per_id[count_row_per_id[target]<4]
        # print(df.shape)
        df_without_id_to_fb=df[df.ID.isin(id_less_than_3_rows.ID)]
        df_with_id_to_fb=df[~df.ID.isin(id_less_than_3_rows.ID)]
        df_with_id_to_fb=df_with_id_to_fb[['date',target,'ID']]
        df_with_id_to_fb.rename({'date':'ds',target:'y'},axis=1,inplace=True)
        
        # print(df.shape)
        # print(id_less_than_3_rows.ID)
        # print(list(id_less_than_3_rows.ID))
        # print(df.groupby('ID').count())
        df_aux=applyParallel_using_df_standar(df_with_id_to_fb,process_to_extract_features_prophet,desc='extracting features from prophet per id',njobs=1)
        print(df_aux.shape)
        df_aux.rename({'ds':'date','y':'target'},axis=1,inplace=True)
        df_aux=pd.concat([df_aux,df_without_id_to_fb])
        print(df_aux.shape)
        print(df_aux.head())
        df_aux.fillna(0,inplace=True)
        # for id in tqdm(df.ID.unique(),desc='extracting features from prophet per id'):
            
            # df_aux=pd.concat([forecast,df_aux])
        df_aux.to_csv(path_features_prophet,index=False)
    return df_aux

def extra_features_manually_no_temporal_variables(df,target='total',D_or_W='D'):
    logging.info('creating features manually no temporal per id')
    path_features_manually_no_temporal=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\02_intermediate\features_manually_no_temporal.csv'
    path_features_manually_no_temporal=modify_path_if_week_periocity(path_features_manually_no_temporal,D_or_W)
    if os.path.isfile(path_features_manually_no_temporal):
        df_features_by_id_no_temporal=pd.read_csv(path_features_manually_no_temporal)
    else:
        #añadir si es casa de vacaciones o no, básicamente si el 95% de los valores son entre junio y agosto y quizá a finales de diciembre hasta el 7 de enero
        df_features_by_id_no_temporal=pd.DataFrame()
        df=df[['ID',target]]
        
        df_features_by_id_no_temporal['percentil_05']=df.groupby("ID")[target].quantile(0.05)
        df_features_by_id_no_temporal['percentil_10']=df.groupby("ID")[target].quantile(0.10)
        df_features_by_id_no_temporal['percentil_25']=df.groupby("ID")[target].quantile(0.25)
        df_features_by_id_no_temporal['percentil_50']=df.groupby("ID")[target].quantile(0.50)
        df_features_by_id_no_temporal['percentil_75']=df.groupby("ID")[target].quantile(0.75)
        df_features_by_id_no_temporal['percentil_90']=df.groupby("ID")[target].quantile(0.90)
        df_features_by_id_no_temporal['percentil_95']=df.groupby("ID")[target].quantile(0.95)
        
        
        df_features_by_id_no_temporal['number_of_peak_2_above_average']= df.groupby("ID")[target].apply(lambda x: x[x>x.quantile(0.5)*2].count())
        df_features_by_id_no_temporal['number_of_peak_over_3_above_average']= df.groupby("ID")[target].apply(lambda x: x[x>x.quantile(0.5)*3].count())
        df_features_by_id_no_temporal['number_of_peak_over_5_above_average']= df.groupby("ID")[target].apply(lambda x: x[x>x.quantile(0.5)*5].count())
        df_features_by_id_no_temporal['number_of_peak_over_10_above_average']= df.groupby("ID")[target].apply(lambda x: x[x>x.quantile(0.5)*10].count())
        df_features_by_id_no_temporal['number_of_peak_over_20_above_average']= df.groupby("ID")[target].apply(lambda x: x[x>x.quantile(0.5)*20].count())
        df_features_by_id_no_temporal['number_of_peak_over_30_above_average']= df.groupby("ID")[target].apply(lambda x: x[x>x.quantile(0.5)*30].count())
        df_features_by_id_no_temporal['number_of_peak_over_50_above_average']= df.groupby("ID")[target].apply(lambda x: x[x>x.quantile(0.5)*50].count())
        df_features_by_id_no_temporal['number_of_peak_over_100_above_average']= df.groupby("ID")[target].apply(lambda x: x[x>x.quantile(0.5)*100].count())

        df_features_by_id_no_temporal['flag_always0']=np.where(df.groupby("ID")[target].sum()==0,True,False)    #añadir si siempre ha sido 0 el valor
        df_features_by_id_no_temporal['maybe_have_something_break']= np.where(
            df.groupby("ID")[target].apply(lambda x: x[x>x.quantile(0.5)*10].count()<=5) ,
            np.where(df.groupby("ID")[target].apply(lambda x: x[x>x.quantile(0.5)*10].count()>0),
                    True,False)
            ,False)
        df_features_by_id_no_temporal.reset_index(inplace=True)
        df_features_by_id_no_temporal.to_csv(path_features_manually_no_temporal,index=False)
    return df_features_by_id_no_temporal

def extra_features_manually_temporal_variables(df,target='total',is_to_predict:bool=False,D_or_W='D'):
    path_features_manually_yes_temporal=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\04_feature\features_manually_yes_temporal.csv'
    path_features_manually_yes_temporal=modify_path_if_week_periocity(path_features_manually_yes_temporal,D_or_W)
    if os.path.isfile(path_features_manually_yes_temporal) and not is_to_predict:
        df_aux=pd.read_csv(path_features_manually_yes_temporal)
        df_aux['date']=pd.to_datetime(df_aux['date'],format='%Y-%m-%d')
    else:
       
        df_aux=pd.DataFrame()
        
        def process_to_extract_features_temporal_variables(df_unique_id,id_unique):
            df_per_id=df[df['ID']==id_unique]
            df_per_id=df_per_id[['ID','date',target]]
                
            df_per_id["1d_roll_sum"] = df_per_id[target].rolling(window=1,closed='left').sum().fillna(method='bfill')
            df_per_id["2d_roll_sum"] = df_per_id[target].rolling(window=2,closed='left').sum().fillna(method='bfill')
            df_per_id["5d_roll_sum"] = df_per_id[target].rolling(window=5,closed='left').sum().fillna(method='bfill')
            df_per_id["7d_roll_sum"] = df_per_id[target].rolling(window=7,closed='left').sum().fillna(method='bfill')
            df_per_id["14d_roll_sum"] = df_per_id[target].rolling(window=14,closed='left').sum().fillna(method='bfill')
            df_per_id["30d_roll_sum"] = df_per_id[target].rolling(window=30,closed='left').sum().fillna(method='bfill')
            df_per_id["60d_roll_sum"] = df_per_id[target].rolling(window=60,closed='left').sum().fillna(method='bfill')
            df_per_id["90d_roll_sum"] = df_per_id[target].rolling(window=90,closed='left').sum().fillna(method='bfill')

            # df_per_id["1d_roll_median"] = df_per_id[target].rolling(window=1,closed='left').median().fillna(method='bfill')
            # df_per_id["2d_roll_median"] = df_per_id[target].rolling(window=2,closed='left').median().fillna(method='bfill')
            # df_per_id["5d_roll_median"] = df_per_id[target].rolling(window=5,closed='left').median().fillna(method='bfill')
            # df_per_id["7d_roll_median"] = df_per_id[target].rolling(window=7,closed='left').median().fillna(method='bfill')
            # df_per_id["14d_roll_median"] = df_per_id[target].rolling(window=14,closed='left').median().fillna(method='bfill')
            # df_per_id["30d_roll_median"] = df_per_id[target].rolling(window=30,closed='left').median().fillna(method='bfill')
            # df_per_id["60d_roll_median"] = df_per_id[target].rolling(window=60,closed='left').median().fillna(method='bfill')
            # df_per_id["90d_roll_median"] = df_per_id[target].rolling(window=90,closed='left').median().fillna(method='bfill')
        
            # df_per_id["1d_roll_mean"] = df_per_id[target].rolling(window=1,closed='left').mean().fillna(method='bfill')
            # df_per_id["2d_roll_mean"] = df_per_id[target].rolling(window=2,closed='left').mean().fillna(method='bfill')
            # df_per_id["5d_roll_mean"] = df_per_id[target].rolling(window=5,closed='left').mean().fillna(method='bfill')
            # df_per_id["7d_roll_mean"] = df_per_id[target].rolling(window=7,closed='left').mean().fillna(method='bfill')
            # df_per_id["14d_roll_mean"] = df_per_id[target].rolling(window=14,closed='left').mean().fillna(method='bfill')
            # df_per_id["30d_roll_mean"] = df_per_id[target].rolling(window=30,closed='left').mean().fillna(method='bfill')
            # df_per_id["60d_roll_mean"] = df_per_id[target].rolling(window=60,closed='left').mean().fillna(method='bfill')
            # df_per_id["90d_roll_mean"] = df_per_id[target].rolling(window=90,closed='left').mean().fillna(method='bfill')

            df_per_id["1d_roll_std"] = df_per_id[target].rolling(window=1,closed='left').std().fillna(method='bfill')
            df_per_id["2d_roll_std"] = df_per_id[target].rolling(window=2,closed='left').std().fillna(method='bfill')
            df_per_id["5d_roll_std"] = df_per_id[target].rolling(window=5,closed='left').std().fillna(method='bfill')
            df_per_id["7d_roll_std"] = df_per_id[target].rolling(window=7,closed='left').std().fillna(method='bfill')
            df_per_id["14d_roll_std"] = df_per_id[target].rolling(window=14,closed='left').std().fillna(method='bfill')
            df_per_id["30d_roll_std"] = df_per_id[target].rolling(window=30,closed='left').std().fillna(method='bfill')
            df_per_id["60d_roll_std"] = df_per_id[target].rolling(window=60,closed='left').std().fillna(method='bfill')
            df_per_id["90d_roll_std"] = df_per_id[target].rolling(window=90,closed='left').std().fillna(method='bfill')
            
            # df_per_id["1d_roll_max"] = df_per_id[target].rolling(window=1,closed='left').max().fillna(method='bfill')
            # df_per_id["2d_roll_max"] = df_per_id[target].rolling(window=2,closed='left').max().fillna(method='bfill')
            # df_per_id["5d_roll_max"] = df_per_id[target].rolling(window=5,closed='left').max().fillna(method='bfill')
            # df_per_id["7d_roll_max"] = df_per_id[target].rolling(window=7,closed='left').max().fillna(method='bfill')
            # df_per_id["14d_roll_max"] = df_per_id[target].rolling(window=14,closed='left').max().fillna(method='bfill')
            # df_per_id["30d_roll_max"] = df_per_id[target].rolling(window=30,closed='left').max().fillna(method='bfill')
            # df_per_id["60d_roll_max"] = df_per_id[target].rolling(window=60,closed='left').max().fillna(method='bfill')
            # df_per_id["90d_roll_max"] = df_per_id[target].rolling(window=90,closed='left').max().fillna(method='bfill')
            
            # df_per_id["1d_roll_min"] = df_per_id[target].rolling(window=1,closed='left').min().fillna(method='bfill')
            # df_per_id["2d_roll_min"] = df_per_id[target].rolling(window=2,closed='left').min().fillna(method='bfill')
            # df_per_id["5d_roll_min"] = df_per_id[target].rolling(window=5,closed='left').min().fillna(method='bfill')
            # df_per_id["7d_roll_min"] = df_per_id[target].rolling(window=7,closed='left').min().fillna(method='bfill')
            # df_per_id["14d_roll_min"] = df_per_id[target].rolling(window=14,closed='left').min().fillna(method='bfill')
            # df_per_id["30d_roll_min"] = df_per_id[target].rolling(window=30,closed='left').min().fillna(method='bfill')
            # df_per_id["60d_roll_min"] = df_per_id[target].rolling(window=60,closed='left').min().fillna(method='bfill')
            # df_per_id["90d_roll_min"] = df_per_id[target].rolling(window=90,closed='left').min().fillna(method='bfill')
        
            df_per_id['shifted_1']=df_per_id[target].shift(periods=1).fillna(method='bfill')
            
            df_per_id['shifted_2']=df_per_id[target].shift(periods=2).fillna(method='bfill')
            df_per_id['shifted_5']=df_per_id[target].shift(periods=5).fillna(method='bfill')
            df_per_id['shifted_7']=df_per_id[target].shift(periods=7).fillna(method='bfill')
            df_per_id['shifted_8']=df_per_id[target].shift(periods=8).fillna(method='bfill')
            df_per_id['shifted_14']=df_per_id[target].shift(periods=14).fillna(method='bfill')
            df_per_id['shifted_diff_1']=df_per_id['shifted_1'].diff(periods=1).fillna(0)
            df_per_id['shifted_diff_2']=df_per_id['shifted_1'].diff(periods=2).fillna(0)
            df_per_id['shifted_diff_5']=df_per_id['shifted_1'].diff(periods=5).fillna(0)
            df_per_id['shifted_diff_7']=df_per_id['shifted_1'].diff(periods=7).fillna(0)
            df_per_id['shifted_diff_8']=df_per_id['shifted_1'].diff(periods=8).fillna(0)
            df_per_id['shifted_diff_14']=df_per_id['shifted_1'].diff(periods=14).fillna(0)
            df_per_id["change_1"] = df_per_id['shifted_1'].div(df_per_id["shifted_1"]).pct_change().mul(100).fillna(0)
            df_per_id["change_2"] = df_per_id['shifted_1'].div(df_per_id["shifted_2"]).pct_change().mul(100).fillna(0)
            df_per_id["change_5"] = df_per_id['shifted_1'].div(df_per_id["shifted_5"]).pct_change().mul(100).fillna(0)
            df_per_id["change_7"] = df_per_id['shifted_1'].div(df_per_id["shifted_7"]).pct_change().mul(100).fillna(0)
            df_per_id["change_8"] = df_per_id['shifted_1'].div(df_per_id["shifted_8"]).pct_change().mul(100).fillna(0)
            df_per_id["change_14"] = df_per_id['shifted_1'].div(df_per_id["shifted_14"]).pct_change().mul(100).fillna(0)
            # df_aux=pd.merge(df_aux,df_per_id,right_on=['ID','date'],left_on=['ID','date'])
            
            df_per_id["running_min"] = df_per_id[target].expanding().min().fillna(0)  # same as cummin()
            df_per_id["running_max"] = df_per_id[target].expanding().max().fillna(0)
            return df_per_id
    

        df_aux=applyParallel_using_df_standar(df,process_to_extract_features_temporal_variables,desc='extracting features temporal manually',njobs=11)
            # if id>2:
            #     return df_aux
        #añadir la suma del mes anterior
        #añadir la suma de la semana anterior
        #añadir la suma del trimestre anterior
        # prin
        df_aux.fillna(0,inplace=True)
        df_aux.replace(np.inf,0,inplace=True)

        df_aux.replace(-np.inf,0,inplace=True)
 
        df_aux.drop(target,inplace=True,axis=1)
        if is_to_predict:
            add_to_the_name='_is_predict.csv'
            path_features_manually_yes_temporal.split('.')[0]+add_to_the_name

        df_aux.to_csv(path_features_manually_yes_temporal,index=False)
    return df_aux

def apply_pca_in_manual_temporal_features(df,components_pca=5,is_to_predict:bool=False,D_or_W='D'):
    
    path_pca_temporal_features=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\04_feature\pca_temporal_features.csv'
    path_pca_temporal_features=modify_path_if_week_periocity(path_pca_temporal_features,D_or_W)
    if os.path.isfile(path_pca_temporal_features) and not is_to_predict:
        X=pd.read_csv(path_pca_temporal_features)
    else:
        if components_pca>0:
            path_pca_temporal_variables=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\06_models\pca_temporal_features.pkl'
            path_pca_temporal_variables=modify_path_if_week_periocity(path_pca_temporal_variables,D_or_W)
            prefix='temporal_variables'
            if os.path.isfile(path_pca_temporal_variables) or is_to_predict :
                to_add_later=df['date']
                X=create_pca_components_without_date(df.drop(['date'],axis=1),path_model=path_pca_temporal_variables,prefix=prefix)                
                X=pd.concat([to_add_later,X],axis=1)
                print(X.columns)
            else:
                
                to_add_later=df['date']
                X=create_pca_components_without_date(df.drop(['date'],axis=1),path_pca_temporal_variables,n_components=5,prefix=prefix)
                X=pd.concat([to_add_later,X],axis=1)
                
        else:
            X=df

        X.to_csv(path_pca_temporal_features,index=False,)
    print(X.head())
    return X
    
def selecting_columns_to_drop():

    corr_075=['total__mean_abs_change',
                'total__median',
                'total__mean',
                'total__length',
                'total__standard_deviation',
                'total__variance',
                'total__skewness',
                'total__kurtosis',
                'total__root_mean_square',
                'total__absolute_sum_of_changes',
                'total__first_location_of_maximum',
                'total__last_location_of_minimum',
                'total__sum_of_reoccurring_values',
                'total__sum_of_reoccurring_data_points',
                'total__ratio_value_number_to_time_series_length',
                'total__sample_entropy',
                'total__maximum',
                'total__absolute_maximum',
                'total__time_reversal_asymmetry_statistic__lag_2',
                'total__time_reversal_asymmetry_statistic__lag_3',
                'total__c3__lag_1',
                'total__c3__lag_2',
                'total__c3__lag_3',
                'total__cid_ce__normalize_False',
                'total__symmetry_looking__r_0.1',
                'total__symmetry_looking__r_0.15000000000000002',
                'total__symmetry_looking__r_0.2',
                'total__symmetry_looking__r_0.25',
                'total__symmetry_looking__r_0.30000000000000004',
                'total__symmetry_looking__r_0.35000000000000003',
                'total__symmetry_looking__r_0.4',
                'total__symmetry_looking__r_0.45',
                'total__symmetry_looking__r_0.5',
                'total__symmetry_looking__r_0.55',
                'total__symmetry_looking__r_0.6000000000000001',
                'total__symmetry_looking__r_0.65',
                'total__symmetry_looking__r_0.7000000000000001',
                'total__symmetry_looking__r_0.75',
                'total__symmetry_looking__r_0.8',
                'total__symmetry_looking__r_0.8500000000000001',
                'total__symmetry_looking__r_0.9',
                'total__symmetry_looking__r_0.9500000000000001',
                'total__large_standard_deviation__r_0.05',
                'total__quantile__q_0.1',
                'total__quantile__q_0.2',
                'total__quantile__q_0.3',
                'total__quantile__q_0.4',
                'total__quantile__q_0.6',
                'total__quantile__q_0.7',
                'total__quantile__q_0.8',
                'total__quantile__q_0.9',
                'total__autocorrelation__lag_2',
                'total__autocorrelation__lag_3',
                'total__autocorrelation__lag_4',
                'total__autocorrelation__lag_5',
                'total__autocorrelation__lag_6',
                'total__autocorrelation__lag_7',
                'total__autocorrelation__lag_8',
                'total__autocorrelation__lag_9',
                'total__agg_autocorrelation__f_agg_"mean"__maxlag_40',
                'total__agg_autocorrelation__f_agg_"median"__maxlag_40',
                'total__agg_autocorrelation__f_agg_"var"__maxlag_40',
                'total__partial_autocorrelation__lag_1',
                'total__number_cwt_peaks__n_1',
                'total__number_cwt_peaks__n_5',
                'total__number_peaks__n_1',
                'total__number_peaks__n_3',
                'total__number_peaks__n_5',
                'total__number_peaks__n_10',
                'total__binned_entropy__max_bins_10',
                'total__index_mass_quantile__q_0.2',
                'total__index_mass_quantile__q_0.3',
                'total__index_mass_quantile__q_0.4',
                'total__index_mass_quantile__q_0.6',
                'total__index_mass_quantile__q_0.7',
                'total__index_mass_quantile__q_0.8',
                'total__index_mass_quantile__q_0.9',
                'total__cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_6__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_6__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_6__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_11__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_11__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_11__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_12__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_12__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_12__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_20__widths_(2, 5, 10, 20)',
                'total__spkt_welch_density__coeff_2',
                'total__spkt_welch_density__coeff_5',
                'total__spkt_welch_density__coeff_8',
                'total__ar_coefficient__coeff_1__k_10',
                'total__ar_coefficient__coeff_2__k_10',
                'total__ar_coefficient__coeff_3__k_10',
                'total__ar_coefficient__coeff_4__k_10',
                'total__ar_coefficient__coeff_5__k_10',
                'total__ar_coefficient__coeff_6__k_10',
                'total__ar_coefficient__coeff_7__k_10',
                'total__ar_coefficient__coeff_8__k_10',
                'total__ar_coefficient__coeff_9__k_10',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.2__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.2__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8',
                'total__fft_coefficient__attr_"real"__coeff_0',
                'total__fft_coefficient__attr_"real"__coeff_1',
                'total__fft_coefficient__attr_"real"__coeff_3',
                'total__fft_coefficient__attr_"real"__coeff_4',
                'total__fft_coefficient__attr_"real"__coeff_5',
                'total__fft_coefficient__attr_"real"__coeff_6',
                'total__fft_coefficient__attr_"real"__coeff_7',
                'total__fft_coefficient__attr_"real"__coeff_8',
                'total__fft_coefficient__attr_"real"__coeff_9',
                'total__fft_coefficient__attr_"real"__coeff_11',
                'total__fft_coefficient__attr_"real"__coeff_15',
                'total__fft_coefficient__attr_"real"__coeff_16',
                'total__fft_coefficient__attr_"real"__coeff_18',
                'total__fft_coefficient__attr_"real"__coeff_21',
                'total__fft_coefficient__attr_"real"__coeff_22',
                'total__fft_coefficient__attr_"real"__coeff_24',
                'total__fft_coefficient__attr_"real"__coeff_25',
                'total__fft_coefficient__attr_"real"__coeff_26',
                'total__fft_coefficient__attr_"real"__coeff_27',
                'total__fft_coefficient__attr_"real"__coeff_28',
                'total__fft_coefficient__attr_"real"__coeff_29',
                'total__fft_coefficient__attr_"real"__coeff_30',
                'total__fft_coefficient__attr_"real"__coeff_32',
                'total__fft_coefficient__attr_"real"__coeff_33',
                'total__fft_coefficient__attr_"real"__coeff_35',
                'total__fft_coefficient__attr_"real"__coeff_37',
                'total__fft_coefficient__attr_"real"__coeff_40',
                'total__fft_coefficient__attr_"real"__coeff_43',
                'total__fft_coefficient__attr_"real"__coeff_45',
                'total__fft_coefficient__attr_"real"__coeff_46',
                'total__fft_coefficient__attr_"real"__coeff_50',
                'total__fft_coefficient__attr_"real"__coeff_51',
                'total__fft_coefficient__attr_"real"__coeff_52',
                'total__fft_coefficient__attr_"real"__coeff_54',
                'total__fft_coefficient__attr_"real"__coeff_57',
                'total__fft_coefficient__attr_"real"__coeff_58',
                'total__fft_coefficient__attr_"real"__coeff_59',
                'total__fft_coefficient__attr_"real"__coeff_60',
                'total__fft_coefficient__attr_"real"__coeff_65',
                'total__fft_coefficient__attr_"real"__coeff_66',
                'total__fft_coefficient__attr_"real"__coeff_67',
                'total__fft_coefficient__attr_"real"__coeff_68',
                'total__fft_coefficient__attr_"real"__coeff_69',
                'total__fft_coefficient__attr_"real"__coeff_72',
                'total__fft_coefficient__attr_"real"__coeff_75',
                'total__fft_coefficient__attr_"real"__coeff_76',
                'total__fft_coefficient__attr_"real"__coeff_78',
                'total__fft_coefficient__attr_"real"__coeff_80',
                'total__fft_coefficient__attr_"real"__coeff_82',
                'total__fft_coefficient__attr_"real"__coeff_83',
                'total__fft_coefficient__attr_"real"__coeff_84',
                'total__fft_coefficient__attr_"real"__coeff_86',
                'total__fft_coefficient__attr_"real"__coeff_87',
                'total__fft_coefficient__attr_"real"__coeff_92',
                'total__fft_coefficient__attr_"real"__coeff_93',
                'total__fft_coefficient__attr_"real"__coeff_98',
                'total__fft_coefficient__attr_"real"__coeff_99',
                'total__fft_coefficient__attr_"imag"__coeff_1',
                'total__fft_coefficient__attr_"imag"__coeff_2',
                'total__fft_coefficient__attr_"imag"__coeff_3',
                'total__fft_coefficient__attr_"imag"__coeff_4',
                'total__fft_coefficient__attr_"imag"__coeff_5',
                'total__fft_coefficient__attr_"imag"__coeff_6',
                'total__fft_coefficient__attr_"imag"__coeff_7',
                'total__fft_coefficient__attr_"imag"__coeff_8',
                'total__fft_coefficient__attr_"imag"__coeff_9',
                'total__fft_coefficient__attr_"imag"__coeff_11',
                'total__fft_coefficient__attr_"imag"__coeff_15',
                'total__fft_coefficient__attr_"imag"__coeff_17',
                'total__fft_coefficient__attr_"imag"__coeff_18',
                'total__fft_coefficient__attr_"imag"__coeff_19',
                'total__fft_coefficient__attr_"imag"__coeff_21',
                'total__fft_coefficient__attr_"imag"__coeff_22',
                'total__fft_coefficient__attr_"imag"__coeff_25',
                'total__fft_coefficient__attr_"imag"__coeff_31',
                'total__fft_coefficient__attr_"imag"__coeff_32',
                'total__fft_coefficient__attr_"imag"__coeff_33',
                'total__fft_coefficient__attr_"imag"__coeff_35',
                'total__fft_coefficient__attr_"imag"__coeff_37',
                'total__fft_coefficient__attr_"imag"__coeff_38',
                'total__fft_coefficient__attr_"imag"__coeff_39',
                'total__fft_coefficient__attr_"imag"__coeff_40',
                'total__fft_coefficient__attr_"imag"__coeff_43',
                'total__fft_coefficient__attr_"imag"__coeff_44',
                'total__fft_coefficient__attr_"imag"__coeff_47',
                'total__fft_coefficient__attr_"imag"__coeff_48',
                'total__fft_coefficient__attr_"imag"__coeff_51',
                'total__fft_coefficient__attr_"imag"__coeff_54',
                'total__fft_coefficient__attr_"imag"__coeff_55',
                'total__fft_coefficient__attr_"imag"__coeff_58',
                'total__fft_coefficient__attr_"imag"__coeff_60',
                'total__fft_coefficient__attr_"imag"__coeff_61',
                'total__fft_coefficient__attr_"imag"__coeff_65',
                'total__fft_coefficient__attr_"imag"__coeff_66',
                'total__fft_coefficient__attr_"imag"__coeff_67',
                'total__fft_coefficient__attr_"imag"__coeff_69',
                'total__fft_coefficient__attr_"imag"__coeff_70',
                'total__fft_coefficient__attr_"imag"__coeff_71',
                'total__fft_coefficient__attr_"imag"__coeff_72',
                'total__fft_coefficient__attr_"imag"__coeff_74',
                'total__fft_coefficient__attr_"imag"__coeff_76',
                'total__fft_coefficient__attr_"imag"__coeff_77',
                'total__fft_coefficient__attr_"imag"__coeff_78',
                'total__fft_coefficient__attr_"imag"__coeff_83',
                'total__fft_coefficient__attr_"imag"__coeff_84',
                'total__fft_coefficient__attr_"imag"__coeff_85',
                'total__fft_coefficient__attr_"imag"__coeff_86',
                'total__fft_coefficient__attr_"imag"__coeff_89',
                'total__fft_coefficient__attr_"imag"__coeff_90',
                'total__fft_coefficient__attr_"imag"__coeff_91',
                'total__fft_coefficient__attr_"imag"__coeff_92',
                'total__fft_coefficient__attr_"imag"__coeff_94',
                'total__fft_coefficient__attr_"imag"__coeff_96',
                'total__fft_coefficient__attr_"imag"__coeff_97',
                'total__fft_coefficient__attr_"imag"__coeff_98',
                'total__fft_coefficient__attr_"imag"__coeff_99',
                'total__fft_coefficient__attr_"abs"__coeff_0',
                'total__fft_coefficient__attr_"abs"__coeff_1',
                'total__fft_coefficient__attr_"abs"__coeff_2',
                'total__fft_coefficient__attr_"abs"__coeff_3',
                'total__fft_coefficient__attr_"abs"__coeff_4',
                'total__fft_coefficient__attr_"abs"__coeff_5',
                'total__fft_coefficient__attr_"abs"__coeff_6',
                'total__fft_coefficient__attr_"abs"__coeff_7',
                'total__fft_coefficient__attr_"abs"__coeff_8',
                'total__fft_coefficient__attr_"abs"__coeff_9',
                'total__fft_coefficient__attr_"abs"__coeff_10',
                'total__fft_coefficient__attr_"abs"__coeff_11',
                'total__fft_coefficient__attr_"abs"__coeff_13',
                'total__fft_coefficient__attr_"abs"__coeff_15',
                'total__fft_coefficient__attr_"abs"__coeff_16',
                'total__fft_coefficient__attr_"abs"__coeff_17',
                'total__fft_coefficient__attr_"abs"__coeff_18',
                'total__fft_coefficient__attr_"abs"__coeff_19',
                'total__fft_coefficient__attr_"abs"__coeff_20',
                'total__fft_coefficient__attr_"abs"__coeff_21',
                'total__fft_coefficient__attr_"abs"__coeff_22',
                'total__fft_coefficient__attr_"abs"__coeff_23',
                'total__fft_coefficient__attr_"abs"__coeff_24',
                'total__fft_coefficient__attr_"abs"__coeff_25',
                'total__fft_coefficient__attr_"abs"__coeff_26',
                'total__fft_coefficient__attr_"abs"__coeff_27',
                'total__fft_coefficient__attr_"abs"__coeff_28',
                'total__fft_coefficient__attr_"abs"__coeff_29',
                'total__fft_coefficient__attr_"abs"__coeff_30',
                'total__fft_coefficient__attr_"abs"__coeff_31',
                'total__fft_coefficient__attr_"abs"__coeff_32',
                'total__fft_coefficient__attr_"abs"__coeff_33',
                'total__fft_coefficient__attr_"abs"__coeff_34',
                'total__fft_coefficient__attr_"abs"__coeff_35',
                'total__fft_coefficient__attr_"abs"__coeff_36',
                'total__fft_coefficient__attr_"abs"__coeff_37',
                'total__fft_coefficient__attr_"abs"__coeff_38',
                'total__fft_coefficient__attr_"abs"__coeff_39',
                'total__fft_coefficient__attr_"abs"__coeff_40',
                'total__fft_coefficient__attr_"abs"__coeff_41',
                'total__fft_coefficient__attr_"abs"__coeff_42',
                'total__fft_coefficient__attr_"abs"__coeff_43',
                'total__fft_coefficient__attr_"abs"__coeff_44',
                'total__fft_coefficient__attr_"abs"__coeff_45',
                'total__fft_coefficient__attr_"abs"__coeff_46',
                'total__fft_coefficient__attr_"abs"__coeff_47',
                'total__fft_coefficient__attr_"abs"__coeff_48',
                'total__fft_coefficient__attr_"abs"__coeff_49',
                'total__fft_coefficient__attr_"abs"__coeff_50',
                'total__fft_coefficient__attr_"abs"__coeff_51',
                'total__fft_coefficient__attr_"abs"__coeff_52',
                'total__fft_coefficient__attr_"abs"__coeff_53',
                'total__fft_coefficient__attr_"abs"__coeff_54',
                'total__fft_coefficient__attr_"abs"__coeff_55',
                'total__fft_coefficient__attr_"abs"__coeff_57',
                'total__fft_coefficient__attr_"abs"__coeff_58',
                'total__fft_coefficient__attr_"abs"__coeff_59',
                'total__fft_coefficient__attr_"abs"__coeff_60',
                'total__fft_coefficient__attr_"abs"__coeff_61',
                'total__fft_coefficient__attr_"abs"__coeff_62',
                'total__fft_coefficient__attr_"abs"__coeff_63',
                'total__fft_coefficient__attr_"abs"__coeff_65',
                'total__fft_coefficient__attr_"abs"__coeff_66',
                'total__fft_coefficient__attr_"abs"__coeff_67',
                'total__fft_coefficient__attr_"abs"__coeff_68',
                'total__fft_coefficient__attr_"abs"__coeff_69',
                'total__fft_coefficient__attr_"abs"__coeff_70',
                'total__fft_coefficient__attr_"abs"__coeff_71',
                'total__fft_coefficient__attr_"abs"__coeff_72',
                'total__fft_coefficient__attr_"abs"__coeff_73',
                'total__fft_coefficient__attr_"abs"__coeff_74',
                'total__fft_coefficient__attr_"abs"__coeff_75',
                'total__fft_coefficient__attr_"abs"__coeff_76',
                'total__fft_coefficient__attr_"abs"__coeff_77',
                'total__fft_coefficient__attr_"abs"__coeff_78',
                'total__fft_coefficient__attr_"abs"__coeff_79',
                'total__fft_coefficient__attr_"abs"__coeff_80',
                'total__fft_coefficient__attr_"abs"__coeff_82',
                'total__fft_coefficient__attr_"abs"__coeff_83',
                'total__fft_coefficient__attr_"abs"__coeff_84',
                'total__fft_coefficient__attr_"abs"__coeff_85',
                'total__fft_coefficient__attr_"abs"__coeff_86',
                'total__fft_coefficient__attr_"abs"__coeff_87',
                'total__fft_coefficient__attr_"abs"__coeff_88',
                'total__fft_coefficient__attr_"abs"__coeff_89',
                'total__fft_coefficient__attr_"abs"__coeff_90',
                'total__fft_coefficient__attr_"abs"__coeff_91',
                'total__fft_coefficient__attr_"abs"__coeff_92',
                'total__fft_coefficient__attr_"abs"__coeff_93',
                'total__fft_coefficient__attr_"abs"__coeff_94',
                'total__fft_coefficient__attr_"abs"__coeff_95',
                'total__fft_coefficient__attr_"abs"__coeff_96',
                'total__fft_coefficient__attr_"abs"__coeff_97',
                'total__fft_coefficient__attr_"abs"__coeff_98',
                'total__fft_coefficient__attr_"abs"__coeff_99',
                'total__fft_coefficient__attr_"angle"__coeff_80',
                'total__fft_aggregated__aggtype_"skew"',
                'total__fft_aggregated__aggtype_"kurtosis"',
                'total__range_count__max_1__min_-1',
                'total__range_count__max_1000000000000.0__min_0',
                'total__approximate_entropy__m_2__r_0.1',
                'total__approximate_entropy__m_2__r_0.3',
                'total__approximate_entropy__m_2__r_0.5',
                'total__approximate_entropy__m_2__r_0.7',
                'total__approximate_entropy__m_2__r_0.9',
                'total__friedrich_coefficients__coeff_3__m_3__r_30',
                'total__max_langevin_fixed_point__m_3__r_30',
                'total__linear_trend__attr_"intercept"',
                'total__linear_trend__attr_"slope"',
                'total__linear_trend__attr_"stderr"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"var"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"min"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"var"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"min"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"var"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"min"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"var"',
                'total__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"',
                'total__number_crossing_m__m_1',
                'total__energy_ratio_by_chunks__num_segments_10__segment_focus_3',
                'total__energy_ratio_by_chunks__num_segments_10__segment_focus_9',
                'total__ratio_beyond_r_sigma__r_0.5',
                'total__ratio_beyond_r_sigma__r_1',
                'total__ratio_beyond_r_sigma__r_1.5',
                'total__ratio_beyond_r_sigma__r_2',
                'total__ratio_beyond_r_sigma__r_2.5',
                'total__ratio_beyond_r_sigma__r_3',
                'total__ratio_beyond_r_sigma__r_10',
                'total__count_below__t_0',
                'total__lempel_ziv_complexity__bins_3',
                'total__lempel_ziv_complexity__bins_5',
                'total__lempel_ziv_complexity__bins_10',
                'total__lempel_ziv_complexity__bins_100',
                'total__fourier_entropy__bins_2',
                'total__fourier_entropy__bins_3',
                'total__fourier_entropy__bins_5',
                'total__fourier_entropy__bins_10',
                'total__fourier_entropy__bins_100',
                'total__permutation_entropy__dimension_3__tau_1',
                'total__permutation_entropy__dimension_4__tau_1',
                'total__permutation_entropy__dimension_5__tau_1',
                'total__permutation_entropy__dimension_6__tau_1',
                'total__permutation_entropy__dimension_7__tau_1',
                'total__matrix_profile__feature_"min"__threshold_0.98',
                'total__matrix_profile__feature_"max"__threshold_0.98',
                'total__matrix_profile__feature_"mean"__threshold_0.98',
                'total__matrix_profile__feature_"median"__threshold_0.98',
                'total__matrix_profile__feature_"25"__threshold_0.98',
                'total__matrix_profile__feature_"75"__threshold_0.98',
                'total__mean_n_absolute_max__number_of_maxima_7',
                'percentil_05',
                'percentil_10',
                'percentil_25',
                'percentil_50',
                'percentil_75',
                'percentil_90',
                'percentil_95',
                'number_of_peak_over_3_above_average',
                'number_of_peak_over_5_above_average',
                'number_of_peak_over_10_above_average',
                'number_of_peak_over_20_above_average',
                'number_of_peak_over_30_above_average',
                'number_of_peak_over_50_above_average',
                'number_of_peak_over_100_above_average',
                'flag_always0',
                '2d_roll_sum',
                '5d_roll_sum',
                '7d_roll_sum',
                '14d_roll_sum',
                '30d_roll_sum',
                '60d_roll_sum',
                '90d_roll_sum',
                '1d_roll_median',
                '2d_roll_median',
                '5d_roll_median',
                '7d_roll_median',
                '14d_roll_median',
                '30d_roll_median',
                '60d_roll_median',
                '90d_roll_median',
                '1d_roll_mean',
                '2d_roll_mean',
                '5d_roll_mean',
                '7d_roll_mean',
                '14d_roll_mean',
                '30d_roll_mean',
                '60d_roll_mean',
                '90d_roll_mean',
                '7d_roll_std',
                '14d_roll_std',
                '60d_roll_std',
                '90d_roll_std',
                '1d_roll_max',
                '2d_roll_max',
                '5d_roll_max',
                '7d_roll_max',
                '14d_roll_max',
                '30d_roll_max',
                '60d_roll_max',
                '90d_roll_max',
                '1d_roll_min',
                '2d_roll_min',
                '5d_roll_min',
                '7d_roll_min',
                '14d_roll_min',
                '30d_roll_min',
                '60d_roll_min',
                '90d_roll_min',
                'shifted_1',
                'shifted_2',
                'running_max']
    
    corr_070=['total__mean_abs_change',
                'total__median',
                'total__mean',
                'total__length',
                'total__standard_deviation',
                'total__variance',
                'total__skewness',
                'total__kurtosis',
                'total__root_mean_square',
                'total__absolute_sum_of_changes',
                'total__first_location_of_maximum',
                'total__last_location_of_minimum',
                'total__sum_of_reoccurring_values',
                'total__sum_of_reoccurring_data_points',
                'total__ratio_value_number_to_time_series_length',
                'total__sample_entropy',
                'total__maximum',
                'total__absolute_maximum',
                'total__time_reversal_asymmetry_statistic__lag_2',
                'total__time_reversal_asymmetry_statistic__lag_3',
                'total__c3__lag_1',
                'total__c3__lag_2',
                'total__c3__lag_3',
                'total__cid_ce__normalize_False',
                'total__symmetry_looking__r_0.1',
                'total__symmetry_looking__r_0.15000000000000002',
                'total__symmetry_looking__r_0.2',
                'total__symmetry_looking__r_0.25',
                'total__symmetry_looking__r_0.30000000000000004',
                'total__symmetry_looking__r_0.35000000000000003',
                'total__symmetry_looking__r_0.4',
                'total__symmetry_looking__r_0.45',
                'total__symmetry_looking__r_0.5',
                'total__symmetry_looking__r_0.55',
                'total__symmetry_looking__r_0.6000000000000001',
                'total__symmetry_looking__r_0.65',
                'total__symmetry_looking__r_0.7000000000000001',
                'total__symmetry_looking__r_0.75',
                'total__symmetry_looking__r_0.8',
                'total__symmetry_looking__r_0.8500000000000001',
                'total__symmetry_looking__r_0.9',
                'total__symmetry_looking__r_0.9500000000000001',
                'total__large_standard_deviation__r_0.05',
                'total__quantile__q_0.1',
                'total__quantile__q_0.2',
                'total__quantile__q_0.3',
                'total__quantile__q_0.4',
                'total__quantile__q_0.6',
                'total__quantile__q_0.7',
                'total__quantile__q_0.8',
                'total__quantile__q_0.9',
                'total__autocorrelation__lag_2',
                'total__autocorrelation__lag_3',
                'total__autocorrelation__lag_4',
                'total__autocorrelation__lag_5',
                'total__autocorrelation__lag_6',
                'total__autocorrelation__lag_7',
                'total__autocorrelation__lag_8',
                'total__autocorrelation__lag_9',
                'total__agg_autocorrelation__f_agg_"mean"__maxlag_40',
                'total__agg_autocorrelation__f_agg_"median"__maxlag_40',
                'total__agg_autocorrelation__f_agg_"var"__maxlag_40',
                'total__partial_autocorrelation__lag_1',
                'total__number_cwt_peaks__n_1',
                'total__number_cwt_peaks__n_5',
                'total__number_peaks__n_1',
                'total__number_peaks__n_3',
                'total__number_peaks__n_5',
                'total__number_peaks__n_10',
                'total__binned_entropy__max_bins_10',
                'total__index_mass_quantile__q_0.2',
                'total__index_mass_quantile__q_0.3',
                'total__index_mass_quantile__q_0.4',
                'total__index_mass_quantile__q_0.6',
                'total__index_mass_quantile__q_0.7',
                'total__index_mass_quantile__q_0.8',
                'total__index_mass_quantile__q_0.9',
                'total__cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_6__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_6__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_6__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_11__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_11__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_11__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_12__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_12__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_12__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_20__widths_(2, 5, 10, 20)',
                'total__spkt_welch_density__coeff_2',
                'total__spkt_welch_density__coeff_5',
                'total__spkt_welch_density__coeff_8',
                'total__ar_coefficient__coeff_1__k_10',
                'total__ar_coefficient__coeff_2__k_10',
                'total__ar_coefficient__coeff_3__k_10',
                'total__ar_coefficient__coeff_4__k_10',
                'total__ar_coefficient__coeff_5__k_10',
                'total__ar_coefficient__coeff_6__k_10',
                'total__ar_coefficient__coeff_7__k_10',
                'total__ar_coefficient__coeff_8__k_10',
                'total__ar_coefficient__coeff_9__k_10',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.2__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.2__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8',
                'total__fft_coefficient__attr_"real"__coeff_0',
                'total__fft_coefficient__attr_"real"__coeff_1',
                'total__fft_coefficient__attr_"real"__coeff_3',
                'total__fft_coefficient__attr_"real"__coeff_4',
                'total__fft_coefficient__attr_"real"__coeff_5',
                'total__fft_coefficient__attr_"real"__coeff_6',
                'total__fft_coefficient__attr_"real"__coeff_7',
                'total__fft_coefficient__attr_"real"__coeff_8',
                'total__fft_coefficient__attr_"real"__coeff_9',
                'total__fft_coefficient__attr_"real"__coeff_11',
                'total__fft_coefficient__attr_"real"__coeff_15',
                'total__fft_coefficient__attr_"real"__coeff_16',
                'total__fft_coefficient__attr_"real"__coeff_18',
                'total__fft_coefficient__attr_"real"__coeff_19',
                'total__fft_coefficient__attr_"real"__coeff_20',
                'total__fft_coefficient__attr_"real"__coeff_21',
                'total__fft_coefficient__attr_"real"__coeff_22',
                'total__fft_coefficient__attr_"real"__coeff_24',
                'total__fft_coefficient__attr_"real"__coeff_25',
                'total__fft_coefficient__attr_"real"__coeff_26',
                'total__fft_coefficient__attr_"real"__coeff_27',
                'total__fft_coefficient__attr_"real"__coeff_28',
                'total__fft_coefficient__attr_"real"__coeff_29',
                'total__fft_coefficient__attr_"real"__coeff_30',
                'total__fft_coefficient__attr_"real"__coeff_31',
                'total__fft_coefficient__attr_"real"__coeff_32',
                'total__fft_coefficient__attr_"real"__coeff_33',
                'total__fft_coefficient__attr_"real"__coeff_35',
                'total__fft_coefficient__attr_"real"__coeff_37',
                'total__fft_coefficient__attr_"real"__coeff_39',
                'total__fft_coefficient__attr_"real"__coeff_40',
                'total__fft_coefficient__attr_"real"__coeff_43',
                'total__fft_coefficient__attr_"real"__coeff_45',
                'total__fft_coefficient__attr_"real"__coeff_46',
                'total__fft_coefficient__attr_"real"__coeff_50',
                'total__fft_coefficient__attr_"real"__coeff_51',
                'total__fft_coefficient__attr_"real"__coeff_52',
                'total__fft_coefficient__attr_"real"__coeff_54',
                'total__fft_coefficient__attr_"real"__coeff_55',
                'total__fft_coefficient__attr_"real"__coeff_57',
                'total__fft_coefficient__attr_"real"__coeff_58',
                'total__fft_coefficient__attr_"real"__coeff_59',
                'total__fft_coefficient__attr_"real"__coeff_60',
                'total__fft_coefficient__attr_"real"__coeff_65',
                'total__fft_coefficient__attr_"real"__coeff_66',
                'total__fft_coefficient__attr_"real"__coeff_67',
                'total__fft_coefficient__attr_"real"__coeff_68',
                'total__fft_coefficient__attr_"real"__coeff_69',
                'total__fft_coefficient__attr_"real"__coeff_72',
                'total__fft_coefficient__attr_"real"__coeff_75',
                'total__fft_coefficient__attr_"real"__coeff_76',
                'total__fft_coefficient__attr_"real"__coeff_77',
                'total__fft_coefficient__attr_"real"__coeff_78',
                'total__fft_coefficient__attr_"real"__coeff_79',
                'total__fft_coefficient__attr_"real"__coeff_80',
                'total__fft_coefficient__attr_"real"__coeff_81',
                'total__fft_coefficient__attr_"real"__coeff_82',
                'total__fft_coefficient__attr_"real"__coeff_83',
                'total__fft_coefficient__attr_"real"__coeff_84',
                'total__fft_coefficient__attr_"real"__coeff_85',
                'total__fft_coefficient__attr_"real"__coeff_86',
                'total__fft_coefficient__attr_"real"__coeff_87',
                'total__fft_coefficient__attr_"real"__coeff_88',
                'total__fft_coefficient__attr_"real"__coeff_89',
                'total__fft_coefficient__attr_"real"__coeff_92',
                'total__fft_coefficient__attr_"real"__coeff_93',
                'total__fft_coefficient__attr_"real"__coeff_98',
                'total__fft_coefficient__attr_"real"__coeff_99',
                'total__fft_coefficient__attr_"imag"__coeff_1',
                'total__fft_coefficient__attr_"imag"__coeff_2',
                'total__fft_coefficient__attr_"imag"__coeff_3',
                'total__fft_coefficient__attr_"imag"__coeff_4',
                'total__fft_coefficient__attr_"imag"__coeff_5',
                'total__fft_coefficient__attr_"imag"__coeff_6',
                'total__fft_coefficient__attr_"imag"__coeff_7',
                'total__fft_coefficient__attr_"imag"__coeff_8',
                'total__fft_coefficient__attr_"imag"__coeff_9',
                'total__fft_coefficient__attr_"imag"__coeff_11',
                'total__fft_coefficient__attr_"imag"__coeff_15',
                'total__fft_coefficient__attr_"imag"__coeff_16',
                'total__fft_coefficient__attr_"imag"__coeff_17',
                'total__fft_coefficient__attr_"imag"__coeff_18',
                'total__fft_coefficient__attr_"imag"__coeff_19',
                'total__fft_coefficient__attr_"imag"__coeff_21',
                'total__fft_coefficient__attr_"imag"__coeff_22',
                'total__fft_coefficient__attr_"imag"__coeff_23',
                'total__fft_coefficient__attr_"imag"__coeff_25',
                'total__fft_coefficient__attr_"imag"__coeff_27',
                'total__fft_coefficient__attr_"imag"__coeff_28',
                'total__fft_coefficient__attr_"imag"__coeff_31',
                'total__fft_coefficient__attr_"imag"__coeff_32',
                'total__fft_coefficient__attr_"imag"__coeff_33',
                'total__fft_coefficient__attr_"imag"__coeff_34',
                'total__fft_coefficient__attr_"imag"__coeff_35',
                'total__fft_coefficient__attr_"imag"__coeff_36',
                'total__fft_coefficient__attr_"imag"__coeff_37',
                'total__fft_coefficient__attr_"imag"__coeff_38',
                'total__fft_coefficient__attr_"imag"__coeff_39',
                'total__fft_coefficient__attr_"imag"__coeff_40',
                'total__fft_coefficient__attr_"imag"__coeff_43',
                'total__fft_coefficient__attr_"imag"__coeff_44',
                'total__fft_coefficient__attr_"imag"__coeff_45',
                'total__fft_coefficient__attr_"imag"__coeff_47',
                'total__fft_coefficient__attr_"imag"__coeff_48',
                'total__fft_coefficient__attr_"imag"__coeff_51',
                'total__fft_coefficient__attr_"imag"__coeff_53',
                'total__fft_coefficient__attr_"imag"__coeff_54',
                'total__fft_coefficient__attr_"imag"__coeff_55',
                'total__fft_coefficient__attr_"imag"__coeff_57',
                'total__fft_coefficient__attr_"imag"__coeff_58',
                'total__fft_coefficient__attr_"imag"__coeff_60',
                'total__fft_coefficient__attr_"imag"__coeff_61',
                'total__fft_coefficient__attr_"imag"__coeff_63',
                'total__fft_coefficient__attr_"imag"__coeff_65',
                'total__fft_coefficient__attr_"imag"__coeff_66',
                'total__fft_coefficient__attr_"imag"__coeff_67',
                'total__fft_coefficient__attr_"imag"__coeff_69',
                'total__fft_coefficient__attr_"imag"__coeff_70',
                'total__fft_coefficient__attr_"imag"__coeff_71',
                'total__fft_coefficient__attr_"imag"__coeff_72',
                'total__fft_coefficient__attr_"imag"__coeff_74',
                'total__fft_coefficient__attr_"imag"__coeff_76',
                'total__fft_coefficient__attr_"imag"__coeff_77',
                'total__fft_coefficient__attr_"imag"__coeff_78',
                'total__fft_coefficient__attr_"imag"__coeff_79',
                'total__fft_coefficient__attr_"imag"__coeff_82',
                'total__fft_coefficient__attr_"imag"__coeff_83',
                'total__fft_coefficient__attr_"imag"__coeff_84',
                'total__fft_coefficient__attr_"imag"__coeff_85',
                'total__fft_coefficient__attr_"imag"__coeff_86',
                'total__fft_coefficient__attr_"imag"__coeff_87',
                'total__fft_coefficient__attr_"imag"__coeff_89',
                'total__fft_coefficient__attr_"imag"__coeff_90',
                'total__fft_coefficient__attr_"imag"__coeff_91',
                'total__fft_coefficient__attr_"imag"__coeff_92',
                'total__fft_coefficient__attr_"imag"__coeff_94',
                'total__fft_coefficient__attr_"imag"__coeff_96',
                'total__fft_coefficient__attr_"imag"__coeff_97',
                'total__fft_coefficient__attr_"imag"__coeff_98',
                'total__fft_coefficient__attr_"imag"__coeff_99',
                'total__fft_coefficient__attr_"abs"__coeff_0',
                'total__fft_coefficient__attr_"abs"__coeff_1',
                'total__fft_coefficient__attr_"abs"__coeff_2',
                'total__fft_coefficient__attr_"abs"__coeff_3',
                'total__fft_coefficient__attr_"abs"__coeff_4',
                'total__fft_coefficient__attr_"abs"__coeff_5',
                'total__fft_coefficient__attr_"abs"__coeff_6',
                'total__fft_coefficient__attr_"abs"__coeff_7',
                'total__fft_coefficient__attr_"abs"__coeff_8',
                'total__fft_coefficient__attr_"abs"__coeff_9',
                'total__fft_coefficient__attr_"abs"__coeff_10',
                'total__fft_coefficient__attr_"abs"__coeff_11',
                'total__fft_coefficient__attr_"abs"__coeff_12',
                'total__fft_coefficient__attr_"abs"__coeff_13',
                'total__fft_coefficient__attr_"abs"__coeff_15',
                'total__fft_coefficient__attr_"abs"__coeff_16',
                'total__fft_coefficient__attr_"abs"__coeff_17',
                'total__fft_coefficient__attr_"abs"__coeff_18',
                'total__fft_coefficient__attr_"abs"__coeff_19',
                'total__fft_coefficient__attr_"abs"__coeff_20',
                'total__fft_coefficient__attr_"abs"__coeff_21',
                'total__fft_coefficient__attr_"abs"__coeff_22',
                'total__fft_coefficient__attr_"abs"__coeff_23',
                'total__fft_coefficient__attr_"abs"__coeff_24',
                'total__fft_coefficient__attr_"abs"__coeff_25',
                'total__fft_coefficient__attr_"abs"__coeff_26',
                'total__fft_coefficient__attr_"abs"__coeff_27',
                'total__fft_coefficient__attr_"abs"__coeff_28',
                'total__fft_coefficient__attr_"abs"__coeff_29',
                'total__fft_coefficient__attr_"abs"__coeff_30',
                'total__fft_coefficient__attr_"abs"__coeff_31',
                'total__fft_coefficient__attr_"abs"__coeff_32',
                'total__fft_coefficient__attr_"abs"__coeff_33',
                'total__fft_coefficient__attr_"abs"__coeff_34',
                'total__fft_coefficient__attr_"abs"__coeff_35',
                'total__fft_coefficient__attr_"abs"__coeff_36',
                'total__fft_coefficient__attr_"abs"__coeff_37',
                'total__fft_coefficient__attr_"abs"__coeff_38',
                'total__fft_coefficient__attr_"abs"__coeff_39',
                'total__fft_coefficient__attr_"abs"__coeff_40',
                'total__fft_coefficient__attr_"abs"__coeff_41',
                'total__fft_coefficient__attr_"abs"__coeff_42',
                'total__fft_coefficient__attr_"abs"__coeff_43',
                'total__fft_coefficient__attr_"abs"__coeff_44',
                'total__fft_coefficient__attr_"abs"__coeff_45',
                'total__fft_coefficient__attr_"abs"__coeff_46',
                'total__fft_coefficient__attr_"abs"__coeff_47',
                'total__fft_coefficient__attr_"abs"__coeff_48',
                'total__fft_coefficient__attr_"abs"__coeff_49',
                'total__fft_coefficient__attr_"abs"__coeff_50',
                'total__fft_coefficient__attr_"abs"__coeff_51',
                'total__fft_coefficient__attr_"abs"__coeff_52',
                'total__fft_coefficient__attr_"abs"__coeff_53',
                'total__fft_coefficient__attr_"abs"__coeff_54',
                'total__fft_coefficient__attr_"abs"__coeff_55',
                'total__fft_coefficient__attr_"abs"__coeff_56',
                'total__fft_coefficient__attr_"abs"__coeff_57',
                'total__fft_coefficient__attr_"abs"__coeff_58',
                'total__fft_coefficient__attr_"abs"__coeff_59',
                'total__fft_coefficient__attr_"abs"__coeff_60',
                'total__fft_coefficient__attr_"abs"__coeff_61',
                'total__fft_coefficient__attr_"abs"__coeff_62',
                'total__fft_coefficient__attr_"abs"__coeff_63',
                'total__fft_coefficient__attr_"abs"__coeff_64',
                'total__fft_coefficient__attr_"abs"__coeff_65',
                'total__fft_coefficient__attr_"abs"__coeff_66',
                'total__fft_coefficient__attr_"abs"__coeff_67',
                'total__fft_coefficient__attr_"abs"__coeff_68',
                'total__fft_coefficient__attr_"abs"__coeff_69',
                'total__fft_coefficient__attr_"abs"__coeff_70',
                'total__fft_coefficient__attr_"abs"__coeff_71',
                'total__fft_coefficient__attr_"abs"__coeff_72',
                'total__fft_coefficient__attr_"abs"__coeff_73',
                'total__fft_coefficient__attr_"abs"__coeff_74',
                'total__fft_coefficient__attr_"abs"__coeff_75',
                'total__fft_coefficient__attr_"abs"__coeff_76',
                'total__fft_coefficient__attr_"abs"__coeff_77',
                'total__fft_coefficient__attr_"abs"__coeff_78',
                'total__fft_coefficient__attr_"abs"__coeff_79',
                'total__fft_coefficient__attr_"abs"__coeff_80',
                'total__fft_coefficient__attr_"abs"__coeff_81',
                'total__fft_coefficient__attr_"abs"__coeff_82',
                'total__fft_coefficient__attr_"abs"__coeff_83',
                'total__fft_coefficient__attr_"abs"__coeff_84',
                'total__fft_coefficient__attr_"abs"__coeff_85',
                'total__fft_coefficient__attr_"abs"__coeff_86',
                'total__fft_coefficient__attr_"abs"__coeff_87',
                'total__fft_coefficient__attr_"abs"__coeff_88',
                'total__fft_coefficient__attr_"abs"__coeff_89',
                'total__fft_coefficient__attr_"abs"__coeff_90',
                'total__fft_coefficient__attr_"abs"__coeff_91',
                'total__fft_coefficient__attr_"abs"__coeff_92',
                'total__fft_coefficient__attr_"abs"__coeff_93',
                'total__fft_coefficient__attr_"abs"__coeff_94',
                'total__fft_coefficient__attr_"abs"__coeff_95',
                'total__fft_coefficient__attr_"abs"__coeff_96',
                'total__fft_coefficient__attr_"abs"__coeff_97',
                'total__fft_coefficient__attr_"abs"__coeff_98',
                'total__fft_coefficient__attr_"abs"__coeff_99',
                'total__fft_coefficient__attr_"angle"__coeff_45',
                'total__fft_coefficient__attr_"angle"__coeff_65',
                'total__fft_coefficient__attr_"angle"__coeff_68',
                'total__fft_coefficient__attr_"angle"__coeff_80',
                'total__fft_coefficient__attr_"angle"__coeff_84',
                'total__fft_coefficient__attr_"angle"__coeff_96',
                'total__fft_aggregated__aggtype_"variance"',
                'total__fft_aggregated__aggtype_"skew"',
                'total__fft_aggregated__aggtype_"kurtosis"',
                'total__value_count__value_0',
                'total__range_count__max_1__min_-1',
                'total__range_count__max_1000000000000.0__min_0',
                'total__approximate_entropy__m_2__r_0.1',
                'total__approximate_entropy__m_2__r_0.3',
                'total__approximate_entropy__m_2__r_0.5',
                'total__approximate_entropy__m_2__r_0.7',
                'total__approximate_entropy__m_2__r_0.9',
                'total__friedrich_coefficients__coeff_3__m_3__r_30',
                'total__max_langevin_fixed_point__m_3__r_30',
                'total__linear_trend__attr_"rvalue"',
                'total__linear_trend__attr_"intercept"',
                'total__linear_trend__attr_"slope"',
                'total__linear_trend__attr_"stderr"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"min"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"var"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"min"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"var"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"min"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"var"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"min"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"var"',
                'total__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"',
                'total__number_crossing_m__m_1',
                'total__energy_ratio_by_chunks__num_segments_10__segment_focus_3',
                'total__energy_ratio_by_chunks__num_segments_10__segment_focus_5',
                'total__energy_ratio_by_chunks__num_segments_10__segment_focus_9',
                'total__ratio_beyond_r_sigma__r_0.5',
                'total__ratio_beyond_r_sigma__r_1',
                'total__ratio_beyond_r_sigma__r_1.5',
                'total__ratio_beyond_r_sigma__r_2',
                'total__ratio_beyond_r_sigma__r_2.5',
                'total__ratio_beyond_r_sigma__r_3',
                'total__ratio_beyond_r_sigma__r_7',
                'total__ratio_beyond_r_sigma__r_10',
                'total__count_below__t_0',
                'total__lempel_ziv_complexity__bins_3',
                'total__lempel_ziv_complexity__bins_5',
                'total__lempel_ziv_complexity__bins_10',
                'total__lempel_ziv_complexity__bins_100',
                'total__fourier_entropy__bins_2',
                'total__fourier_entropy__bins_3',
                'total__fourier_entropy__bins_5',
                'total__fourier_entropy__bins_10',
                'total__fourier_entropy__bins_100',
                'total__permutation_entropy__dimension_3__tau_1',
                'total__permutation_entropy__dimension_4__tau_1',
                'total__permutation_entropy__dimension_5__tau_1',
                'total__permutation_entropy__dimension_6__tau_1',
                'total__permutation_entropy__dimension_7__tau_1',
                'total__matrix_profile__feature_"min"__threshold_0.98',
                'total__matrix_profile__feature_"max"__threshold_0.98',
                'total__matrix_profile__feature_"mean"__threshold_0.98',
                'total__matrix_profile__feature_"median"__threshold_0.98',
                'total__matrix_profile__feature_"25"__threshold_0.98',
                'total__matrix_profile__feature_"75"__threshold_0.98',
                'total__mean_n_absolute_max__number_of_maxima_7',
                'percentil_05',
                'percentil_10',
                'percentil_25',
                'percentil_50',
                'percentil_75',
                'percentil_90',
                'percentil_95',
                'number_of_peak_2_above_average',
                'number_of_peak_over_3_above_average',
                'number_of_peak_over_5_above_average',
                'number_of_peak_over_10_above_average',
                'number_of_peak_over_20_above_average',
                'number_of_peak_over_30_above_average',
                'number_of_peak_over_50_above_average',
                'number_of_peak_over_100_above_average',
                'flag_always0',
                '2d_roll_sum',
                '5d_roll_sum',
                '7d_roll_sum',
                '14d_roll_sum',
                '30d_roll_sum',
                '60d_roll_sum',
                '90d_roll_sum',
                '1d_roll_median',
                '2d_roll_median',
                '5d_roll_median',
                '7d_roll_median',
                '14d_roll_median',
                '30d_roll_median',
                '60d_roll_median',
                '90d_roll_median',
                '1d_roll_mean',
                '2d_roll_mean',
                '5d_roll_mean',
                '7d_roll_mean',
                '14d_roll_mean',
                '30d_roll_mean',
                '60d_roll_mean',
                '90d_roll_mean',
                '7d_roll_std',
                '14d_roll_std',
                '30d_roll_std',
                '60d_roll_std',
                '90d_roll_std',
                '1d_roll_max',
                '2d_roll_max',
                '5d_roll_max',
                '7d_roll_max',
                '14d_roll_max',
                '30d_roll_max',
                '60d_roll_max',
                '90d_roll_max',
                '1d_roll_min',
                '2d_roll_min',
                '5d_roll_min',
                '7d_roll_min',
                '14d_roll_min',
                '30d_roll_min',
                '60d_roll_min',
                '90d_roll_min',
                'shifted_1',
                'shifted_2',
                'shifted_7',
                'shifted_14',
                'running_max']
    
    corr_065=['total__abs_energy',
                'total__mean_abs_change',
                'total__median',
                'total__mean',
                'total__length',
                'total__standard_deviation',
                'total__variance',
                'total__skewness',
                'total__kurtosis',
                'total__root_mean_square',
                'total__absolute_sum_of_changes',
                'total__count_below_mean',
                'total__first_location_of_maximum',
                'total__last_location_of_minimum',
                'total__percentage_of_reoccurring_values_to_all_values',
                'total__sum_of_reoccurring_values',
                'total__sum_of_reoccurring_data_points',
                'total__ratio_value_number_to_time_series_length',
                'total__sample_entropy',
                'total__maximum',
                'total__absolute_maximum',
                'total__time_reversal_asymmetry_statistic__lag_2',
                'total__time_reversal_asymmetry_statistic__lag_3',
                'total__c3__lag_1',
                'total__c3__lag_2',
                'total__c3__lag_3',
                'total__cid_ce__normalize_False',
                'total__symmetry_looking__r_0.1',
                'total__symmetry_looking__r_0.15000000000000002',
                'total__symmetry_looking__r_0.2',
                'total__symmetry_looking__r_0.25',
                'total__symmetry_looking__r_0.30000000000000004',
                'total__symmetry_looking__r_0.35000000000000003',
                'total__symmetry_looking__r_0.4',
                'total__symmetry_looking__r_0.45',
                'total__symmetry_looking__r_0.5',
                'total__symmetry_looking__r_0.55',
                'total__symmetry_looking__r_0.6000000000000001',
                'total__symmetry_looking__r_0.65',
                'total__symmetry_looking__r_0.7000000000000001',
                'total__symmetry_looking__r_0.75',
                'total__symmetry_looking__r_0.8',
                'total__symmetry_looking__r_0.8500000000000001',
                'total__symmetry_looking__r_0.9',
                'total__symmetry_looking__r_0.9500000000000001',
                'total__large_standard_deviation__r_0.05',
                'total__large_standard_deviation__r_0.2',
                'total__quantile__q_0.1',
                'total__quantile__q_0.2',
                'total__quantile__q_0.3',
                'total__quantile__q_0.4',
                'total__quantile__q_0.6',
                'total__quantile__q_0.7',
                'total__quantile__q_0.8',
                'total__quantile__q_0.9',
                'total__autocorrelation__lag_1',
                'total__autocorrelation__lag_2',
                'total__autocorrelation__lag_3',
                'total__autocorrelation__lag_4',
                'total__autocorrelation__lag_5',
                'total__autocorrelation__lag_6',
                'total__autocorrelation__lag_7',
                'total__autocorrelation__lag_8',
                'total__autocorrelation__lag_9',
                'total__agg_autocorrelation__f_agg_"mean"__maxlag_40',
                'total__agg_autocorrelation__f_agg_"median"__maxlag_40',
                'total__agg_autocorrelation__f_agg_"var"__maxlag_40',
                'total__partial_autocorrelation__lag_1',
                'total__partial_autocorrelation__lag_2',
                'total__number_cwt_peaks__n_1',
                'total__number_cwt_peaks__n_5',
                'total__number_peaks__n_1',
                'total__number_peaks__n_3',
                'total__number_peaks__n_5',
                'total__number_peaks__n_10',
                'total__binned_entropy__max_bins_10',
                'total__index_mass_quantile__q_0.2',
                'total__index_mass_quantile__q_0.3',
                'total__index_mass_quantile__q_0.4',
                'total__index_mass_quantile__q_0.6',
                'total__index_mass_quantile__q_0.7',
                'total__index_mass_quantile__q_0.8',
                'total__index_mass_quantile__q_0.9',
                'total__cwt_coefficients__coeff_0__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_1__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_2__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_3__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_4__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_5__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_6__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_6__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_6__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_7__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_8__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_9__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_10__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_11__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_11__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_11__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_12__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_12__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_12__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_12__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_13__w_20__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_2__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_5__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_10__widths_(2, 5, 10, 20)',
                'total__cwt_coefficients__coeff_14__w_20__widths_(2, 5, 10, 20)',
                'total__spkt_welch_density__coeff_2',
                'total__spkt_welch_density__coeff_5',
                'total__spkt_welch_density__coeff_8',
                'total__ar_coefficient__coeff_0__k_10',
                'total__ar_coefficient__coeff_1__k_10',
                'total__ar_coefficient__coeff_2__k_10',
                'total__ar_coefficient__coeff_3__k_10',
                'total__ar_coefficient__coeff_4__k_10',
                'total__ar_coefficient__coeff_5__k_10',
                'total__ar_coefficient__coeff_6__k_10',
                'total__ar_coefficient__coeff_7__k_10',
                'total__ar_coefficient__coeff_8__k_10',
                'total__ar_coefficient__coeff_9__k_10',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.2__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.2__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.2__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.0',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.4__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.4__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.2',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.2',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.6__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.6__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.6__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_0.8__ql_0.6',
                'total__change_quantiles__f_agg_"mean"__isabs_False__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
                'total__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8',
                'total__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
                'total__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8',
                'total__fft_coefficient__attr_"real"__coeff_0',
                'total__fft_coefficient__attr_"real"__coeff_1',
                'total__fft_coefficient__attr_"real"__coeff_3',
                'total__fft_coefficient__attr_"real"__coeff_4',
                'total__fft_coefficient__attr_"real"__coeff_5',
                'total__fft_coefficient__attr_"real"__coeff_6',
                'total__fft_coefficient__attr_"real"__coeff_7',
                'total__fft_coefficient__attr_"real"__coeff_8',
                'total__fft_coefficient__attr_"real"__coeff_9',
                'total__fft_coefficient__attr_"real"__coeff_10',
                'total__fft_coefficient__attr_"real"__coeff_11',
                'total__fft_coefficient__attr_"real"__coeff_15',
                'total__fft_coefficient__attr_"real"__coeff_16',
                'total__fft_coefficient__attr_"real"__coeff_18',
                'total__fft_coefficient__attr_"real"__coeff_19',
                'total__fft_coefficient__attr_"real"__coeff_20',
                'total__fft_coefficient__attr_"real"__coeff_21',
                'total__fft_coefficient__attr_"real"__coeff_22',
                'total__fft_coefficient__attr_"real"__coeff_24',
                'total__fft_coefficient__attr_"real"__coeff_25',
                'total__fft_coefficient__attr_"real"__coeff_26',
                'total__fft_coefficient__attr_"real"__coeff_27',
                'total__fft_coefficient__attr_"real"__coeff_28',
                'total__fft_coefficient__attr_"real"__coeff_29',
                'total__fft_coefficient__attr_"real"__coeff_30',
                'total__fft_coefficient__attr_"real"__coeff_31',
                'total__fft_coefficient__attr_"real"__coeff_32',
                'total__fft_coefficient__attr_"real"__coeff_33',
                'total__fft_coefficient__attr_"real"__coeff_35',
                'total__fft_coefficient__attr_"real"__coeff_37',
                'total__fft_coefficient__attr_"real"__coeff_39',
                'total__fft_coefficient__attr_"real"__coeff_40',
                'total__fft_coefficient__attr_"real"__coeff_41',
                'total__fft_coefficient__attr_"real"__coeff_43',
                'total__fft_coefficient__attr_"real"__coeff_45',
                'total__fft_coefficient__attr_"real"__coeff_46',
                'total__fft_coefficient__attr_"real"__coeff_50',
                'total__fft_coefficient__attr_"real"__coeff_51',
                'total__fft_coefficient__attr_"real"__coeff_52',
                'total__fft_coefficient__attr_"real"__coeff_54',
                'total__fft_coefficient__attr_"real"__coeff_55',
                'total__fft_coefficient__attr_"real"__coeff_57',
                'total__fft_coefficient__attr_"real"__coeff_58',
                'total__fft_coefficient__attr_"real"__coeff_59',
                'total__fft_coefficient__attr_"real"__coeff_60',
                'total__fft_coefficient__attr_"real"__coeff_65',
                'total__fft_coefficient__attr_"real"__coeff_66',
                'total__fft_coefficient__attr_"real"__coeff_67',
                'total__fft_coefficient__attr_"real"__coeff_68',
                'total__fft_coefficient__attr_"real"__coeff_69',
                'total__fft_coefficient__attr_"real"__coeff_71',
                'total__fft_coefficient__attr_"real"__coeff_72',
                'total__fft_coefficient__attr_"real"__coeff_75',
                'total__fft_coefficient__attr_"real"__coeff_76',
                'total__fft_coefficient__attr_"real"__coeff_77',
                'total__fft_coefficient__attr_"real"__coeff_78',
                'total__fft_coefficient__attr_"real"__coeff_79',
                'total__fft_coefficient__attr_"real"__coeff_80',
                'total__fft_coefficient__attr_"real"__coeff_81',
                'total__fft_coefficient__attr_"real"__coeff_82',
                'total__fft_coefficient__attr_"real"__coeff_83',
                'total__fft_coefficient__attr_"real"__coeff_84',
                'total__fft_coefficient__attr_"real"__coeff_85',
                'total__fft_coefficient__attr_"real"__coeff_86',
                'total__fft_coefficient__attr_"real"__coeff_87',
                'total__fft_coefficient__attr_"real"__coeff_88',
                'total__fft_coefficient__attr_"real"__coeff_89',
                'total__fft_coefficient__attr_"real"__coeff_91',
                'total__fft_coefficient__attr_"real"__coeff_92',
                'total__fft_coefficient__attr_"real"__coeff_93',
                'total__fft_coefficient__attr_"real"__coeff_97',
                'total__fft_coefficient__attr_"real"__coeff_98',
                'total__fft_coefficient__attr_"real"__coeff_99',
                'total__fft_coefficient__attr_"imag"__coeff_1',
                'total__fft_coefficient__attr_"imag"__coeff_2',
                'total__fft_coefficient__attr_"imag"__coeff_3',
                'total__fft_coefficient__attr_"imag"__coeff_4',
                'total__fft_coefficient__attr_"imag"__coeff_5',
                'total__fft_coefficient__attr_"imag"__coeff_6',
                'total__fft_coefficient__attr_"imag"__coeff_7',
                'total__fft_coefficient__attr_"imag"__coeff_8',
                'total__fft_coefficient__attr_"imag"__coeff_9',
                'total__fft_coefficient__attr_"imag"__coeff_11',
                'total__fft_coefficient__attr_"imag"__coeff_14',
                'total__fft_coefficient__attr_"imag"__coeff_15',
                'total__fft_coefficient__attr_"imag"__coeff_16',
                'total__fft_coefficient__attr_"imag"__coeff_17',
                'total__fft_coefficient__attr_"imag"__coeff_18',
                'total__fft_coefficient__attr_"imag"__coeff_19',
                'total__fft_coefficient__attr_"imag"__coeff_20',
                'total__fft_coefficient__attr_"imag"__coeff_21',
                'total__fft_coefficient__attr_"imag"__coeff_22',
                'total__fft_coefficient__attr_"imag"__coeff_23',
                'total__fft_coefficient__attr_"imag"__coeff_25',
                'total__fft_coefficient__attr_"imag"__coeff_26',
                'total__fft_coefficient__attr_"imag"__coeff_27',
                'total__fft_coefficient__attr_"imag"__coeff_28',
                'total__fft_coefficient__attr_"imag"__coeff_31',
                'total__fft_coefficient__attr_"imag"__coeff_32',
                'total__fft_coefficient__attr_"imag"__coeff_33',
                'total__fft_coefficient__attr_"imag"__coeff_34',
                'total__fft_coefficient__attr_"imag"__coeff_35',
                'total__fft_coefficient__attr_"imag"__coeff_36',
                'total__fft_coefficient__attr_"imag"__coeff_37',
                'total__fft_coefficient__attr_"imag"__coeff_38',
                'total__fft_coefficient__attr_"imag"__coeff_39',
                'total__fft_coefficient__attr_"imag"__coeff_40',
                'total__fft_coefficient__attr_"imag"__coeff_43',
                'total__fft_coefficient__attr_"imag"__coeff_44',
                'total__fft_coefficient__attr_"imag"__coeff_45',
                'total__fft_coefficient__attr_"imag"__coeff_47',
                'total__fft_coefficient__attr_"imag"__coeff_48',
                'total__fft_coefficient__attr_"imag"__coeff_49',
                'total__fft_coefficient__attr_"imag"__coeff_50',
                'total__fft_coefficient__attr_"imag"__coeff_51',
                'total__fft_coefficient__attr_"imag"__coeff_52',
                'total__fft_coefficient__attr_"imag"__coeff_53',
                'total__fft_coefficient__attr_"imag"__coeff_54',
                'total__fft_coefficient__attr_"imag"__coeff_55',
                'total__fft_coefficient__attr_"imag"__coeff_57',
                'total__fft_coefficient__attr_"imag"__coeff_58',
                'total__fft_coefficient__attr_"imag"__coeff_60',
                'total__fft_coefficient__attr_"imag"__coeff_61',
                'total__fft_coefficient__attr_"imag"__coeff_62',
                'total__fft_coefficient__attr_"imag"__coeff_63',
                'total__fft_coefficient__attr_"imag"__coeff_64',
                'total__fft_coefficient__attr_"imag"__coeff_65',
                'total__fft_coefficient__attr_"imag"__coeff_66',
                'total__fft_coefficient__attr_"imag"__coeff_67',
                'total__fft_coefficient__attr_"imag"__coeff_69',
                'total__fft_coefficient__attr_"imag"__coeff_70',
                'total__fft_coefficient__attr_"imag"__coeff_71',
                'total__fft_coefficient__attr_"imag"__coeff_72',
                'total__fft_coefficient__attr_"imag"__coeff_74',
                'total__fft_coefficient__attr_"imag"__coeff_76',
                'total__fft_coefficient__attr_"imag"__coeff_77',
                'total__fft_coefficient__attr_"imag"__coeff_78',
                'total__fft_coefficient__attr_"imag"__coeff_79',
                'total__fft_coefficient__attr_"imag"__coeff_82',
                'total__fft_coefficient__attr_"imag"__coeff_83',
                'total__fft_coefficient__attr_"imag"__coeff_84',
                'total__fft_coefficient__attr_"imag"__coeff_85',
                'total__fft_coefficient__attr_"imag"__coeff_86',
                'total__fft_coefficient__attr_"imag"__coeff_87',
                'total__fft_coefficient__attr_"imag"__coeff_89',
                'total__fft_coefficient__attr_"imag"__coeff_90',
                'total__fft_coefficient__attr_"imag"__coeff_91',
                'total__fft_coefficient__attr_"imag"__coeff_92',
                'total__fft_coefficient__attr_"imag"__coeff_93',
                'total__fft_coefficient__attr_"imag"__coeff_94',
                'total__fft_coefficient__attr_"imag"__coeff_95',
                'total__fft_coefficient__attr_"imag"__coeff_96',
                'total__fft_coefficient__attr_"imag"__coeff_97',
                'total__fft_coefficient__attr_"imag"__coeff_98',
                'total__fft_coefficient__attr_"imag"__coeff_99',
                'total__fft_coefficient__attr_"abs"__coeff_0',
                'total__fft_coefficient__attr_"abs"__coeff_1',
                'total__fft_coefficient__attr_"abs"__coeff_2',
                'total__fft_coefficient__attr_"abs"__coeff_3',
                'total__fft_coefficient__attr_"abs"__coeff_4',
                'total__fft_coefficient__attr_"abs"__coeff_5',
                'total__fft_coefficient__attr_"abs"__coeff_6',
                'total__fft_coefficient__attr_"abs"__coeff_7',
                'total__fft_coefficient__attr_"abs"__coeff_8',
                'total__fft_coefficient__attr_"abs"__coeff_9',
                'total__fft_coefficient__attr_"abs"__coeff_10',
                'total__fft_coefficient__attr_"abs"__coeff_11',
                'total__fft_coefficient__attr_"abs"__coeff_12',
                'total__fft_coefficient__attr_"abs"__coeff_13',
                'total__fft_coefficient__attr_"abs"__coeff_14',
                'total__fft_coefficient__attr_"abs"__coeff_15',
                'total__fft_coefficient__attr_"abs"__coeff_16',
                'total__fft_coefficient__attr_"abs"__coeff_17',
                'total__fft_coefficient__attr_"abs"__coeff_18',
                'total__fft_coefficient__attr_"abs"__coeff_19',
                'total__fft_coefficient__attr_"abs"__coeff_20',
                'total__fft_coefficient__attr_"abs"__coeff_21',
                'total__fft_coefficient__attr_"abs"__coeff_22',
                'total__fft_coefficient__attr_"abs"__coeff_23',
                'total__fft_coefficient__attr_"abs"__coeff_24',
                'total__fft_coefficient__attr_"abs"__coeff_25',
                'total__fft_coefficient__attr_"abs"__coeff_26',
                'total__fft_coefficient__attr_"abs"__coeff_27',
                'total__fft_coefficient__attr_"abs"__coeff_28',
                'total__fft_coefficient__attr_"abs"__coeff_29',
                'total__fft_coefficient__attr_"abs"__coeff_30',
                'total__fft_coefficient__attr_"abs"__coeff_31',
                'total__fft_coefficient__attr_"abs"__coeff_32',
                'total__fft_coefficient__attr_"abs"__coeff_33',
                'total__fft_coefficient__attr_"abs"__coeff_34',
                'total__fft_coefficient__attr_"abs"__coeff_35',
                'total__fft_coefficient__attr_"abs"__coeff_36',
                'total__fft_coefficient__attr_"abs"__coeff_37',
                'total__fft_coefficient__attr_"abs"__coeff_38',
                'total__fft_coefficient__attr_"abs"__coeff_39',
                'total__fft_coefficient__attr_"abs"__coeff_40',
                'total__fft_coefficient__attr_"abs"__coeff_41',
                'total__fft_coefficient__attr_"abs"__coeff_42',
                'total__fft_coefficient__attr_"abs"__coeff_43',
                'total__fft_coefficient__attr_"abs"__coeff_44',
                'total__fft_coefficient__attr_"abs"__coeff_45',
                'total__fft_coefficient__attr_"abs"__coeff_46',
                'total__fft_coefficient__attr_"abs"__coeff_47',
                'total__fft_coefficient__attr_"abs"__coeff_48',
                'total__fft_coefficient__attr_"abs"__coeff_49',
                'total__fft_coefficient__attr_"abs"__coeff_50',
                'total__fft_coefficient__attr_"abs"__coeff_51',
                'total__fft_coefficient__attr_"abs"__coeff_52',
                'total__fft_coefficient__attr_"abs"__coeff_53',
                'total__fft_coefficient__attr_"abs"__coeff_54',
                'total__fft_coefficient__attr_"abs"__coeff_55',
                'total__fft_coefficient__attr_"abs"__coeff_56',
                'total__fft_coefficient__attr_"abs"__coeff_57',
                'total__fft_coefficient__attr_"abs"__coeff_58',
                'total__fft_coefficient__attr_"abs"__coeff_59',
                'total__fft_coefficient__attr_"abs"__coeff_60',
                'total__fft_coefficient__attr_"abs"__coeff_61',
                'total__fft_coefficient__attr_"abs"__coeff_62',
                'total__fft_coefficient__attr_"abs"__coeff_63',
                'total__fft_coefficient__attr_"abs"__coeff_64',
                'total__fft_coefficient__attr_"abs"__coeff_65',
                'total__fft_coefficient__attr_"abs"__coeff_66',
                'total__fft_coefficient__attr_"abs"__coeff_67',
                'total__fft_coefficient__attr_"abs"__coeff_68',
                'total__fft_coefficient__attr_"abs"__coeff_69',
                'total__fft_coefficient__attr_"abs"__coeff_70',
                'total__fft_coefficient__attr_"abs"__coeff_71',
                'total__fft_coefficient__attr_"abs"__coeff_72',
                'total__fft_coefficient__attr_"abs"__coeff_73',
                'total__fft_coefficient__attr_"abs"__coeff_74',
                'total__fft_coefficient__attr_"abs"__coeff_75',
                'total__fft_coefficient__attr_"abs"__coeff_76',
                'total__fft_coefficient__attr_"abs"__coeff_77',
                'total__fft_coefficient__attr_"abs"__coeff_78',
                'total__fft_coefficient__attr_"abs"__coeff_79',
                'total__fft_coefficient__attr_"abs"__coeff_80',
                'total__fft_coefficient__attr_"abs"__coeff_81',
                'total__fft_coefficient__attr_"abs"__coeff_82',
                'total__fft_coefficient__attr_"abs"__coeff_83',
                'total__fft_coefficient__attr_"abs"__coeff_84',
                'total__fft_coefficient__attr_"abs"__coeff_85',
                'total__fft_coefficient__attr_"abs"__coeff_86',
                'total__fft_coefficient__attr_"abs"__coeff_87',
                'total__fft_coefficient__attr_"abs"__coeff_88',
                'total__fft_coefficient__attr_"abs"__coeff_89',
                'total__fft_coefficient__attr_"abs"__coeff_90',
                'total__fft_coefficient__attr_"abs"__coeff_91',
                'total__fft_coefficient__attr_"abs"__coeff_92',
                'total__fft_coefficient__attr_"abs"__coeff_93',
                'total__fft_coefficient__attr_"abs"__coeff_94',
                'total__fft_coefficient__attr_"abs"__coeff_95',
                'total__fft_coefficient__attr_"abs"__coeff_96',
                'total__fft_coefficient__attr_"abs"__coeff_97',
                'total__fft_coefficient__attr_"abs"__coeff_98',
                'total__fft_coefficient__attr_"abs"__coeff_99',
                'total__fft_coefficient__attr_"angle"__coeff_4',
                'total__fft_coefficient__attr_"angle"__coeff_10',
                'total__fft_coefficient__attr_"angle"__coeff_13',
                'total__fft_coefficient__attr_"angle"__coeff_20',
                'total__fft_coefficient__attr_"angle"__coeff_23',
                'total__fft_coefficient__attr_"angle"__coeff_26',
                'total__fft_coefficient__attr_"angle"__coeff_28',
                'total__fft_coefficient__attr_"angle"__coeff_42',
                'total__fft_coefficient__attr_"angle"__coeff_44',
                'total__fft_coefficient__attr_"angle"__coeff_45',
                'total__fft_coefficient__attr_"angle"__coeff_46',
                'total__fft_coefficient__attr_"angle"__coeff_52',
                'total__fft_coefficient__attr_"angle"__coeff_59',
                'total__fft_coefficient__attr_"angle"__coeff_65',
                'total__fft_coefficient__attr_"angle"__coeff_66',
                'total__fft_coefficient__attr_"angle"__coeff_67',
                'total__fft_coefficient__attr_"angle"__coeff_68',
                'total__fft_coefficient__attr_"angle"__coeff_71',
                'total__fft_coefficient__attr_"angle"__coeff_73',
                'total__fft_coefficient__attr_"angle"__coeff_75',
                'total__fft_coefficient__attr_"angle"__coeff_76',
                'total__fft_coefficient__attr_"angle"__coeff_80',
                'total__fft_coefficient__attr_"angle"__coeff_82',
                'total__fft_coefficient__attr_"angle"__coeff_84',
                'total__fft_coefficient__attr_"angle"__coeff_87',
                'total__fft_coefficient__attr_"angle"__coeff_96',
                'total__fft_coefficient__attr_"angle"__coeff_98',
                'total__fft_aggregated__aggtype_"centroid"',
                'total__fft_aggregated__aggtype_"variance"',
                'total__fft_aggregated__aggtype_"skew"',
                'total__fft_aggregated__aggtype_"kurtosis"',
                'total__value_count__value_0',
                'total__value_count__value_1',
                'total__range_count__max_1__min_-1',
                'total__range_count__max_1000000000000.0__min_0',
                'total__approximate_entropy__m_2__r_0.1',
                'total__approximate_entropy__m_2__r_0.3',
                'total__approximate_entropy__m_2__r_0.5',
                'total__approximate_entropy__m_2__r_0.7',
                'total__approximate_entropy__m_2__r_0.9',
                'total__friedrich_coefficients__coeff_3__m_3__r_30',
                'total__max_langevin_fixed_point__m_3__r_30',
                'total__linear_trend__attr_"rvalue"',
                'total__linear_trend__attr_"intercept"',
                'total__linear_trend__attr_"slope"',
                'total__linear_trend__attr_"stderr"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"min"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"rvalue"__chunk_len_50__f_agg_"var"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"min"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"intercept"__chunk_len_50__f_agg_"var"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"min"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"slope"__chunk_len_50__f_agg_"var"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"max"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"min"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"mean"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_5__f_agg_"var"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"max"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"min"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"mean"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_10__f_agg_"var"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"max"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"min"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"mean"',
                'total__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"var"',
                'total__augmented_dickey_fuller__attr_"teststat"__autolag_"AIC"',
                'total__augmented_dickey_fuller__attr_"pvalue"__autolag_"AIC"',
                'total__number_crossing_m__m_1',
                'total__energy_ratio_by_chunks__num_segments_10__segment_focus_1',
                'total__energy_ratio_by_chunks__num_segments_10__segment_focus_3',
                'total__energy_ratio_by_chunks__num_segments_10__segment_focus_5',
                'total__energy_ratio_by_chunks__num_segments_10__segment_focus_6',
                'total__energy_ratio_by_chunks__num_segments_10__segment_focus_8',
                'total__energy_ratio_by_chunks__num_segments_10__segment_focus_9',
                'total__ratio_beyond_r_sigma__r_0.5',
                'total__ratio_beyond_r_sigma__r_1',
                'total__ratio_beyond_r_sigma__r_1.5',
                'total__ratio_beyond_r_sigma__r_2',
                'total__ratio_beyond_r_sigma__r_2.5',
                'total__ratio_beyond_r_sigma__r_3',
                'total__ratio_beyond_r_sigma__r_7',
                'total__ratio_beyond_r_sigma__r_10',
                'total__count_below__t_0',
                'total__lempel_ziv_complexity__bins_2',
                'total__lempel_ziv_complexity__bins_3',
                'total__lempel_ziv_complexity__bins_5',
                'total__lempel_ziv_complexity__bins_10',
                'total__lempel_ziv_complexity__bins_100',
                'total__fourier_entropy__bins_2',
                'total__fourier_entropy__bins_3',
                'total__fourier_entropy__bins_5',
                'total__fourier_entropy__bins_10',
                'total__fourier_entropy__bins_100',
                'total__permutation_entropy__dimension_3__tau_1',
                'total__permutation_entropy__dimension_4__tau_1',
                'total__permutation_entropy__dimension_5__tau_1',
                'total__permutation_entropy__dimension_6__tau_1',
                'total__permutation_entropy__dimension_7__tau_1',
                'total__matrix_profile__feature_"min"__threshold_0.98',
                'total__matrix_profile__feature_"max"__threshold_0.98',
                'total__matrix_profile__feature_"mean"__threshold_0.98',
                'total__matrix_profile__feature_"median"__threshold_0.98',
                'total__matrix_profile__feature_"25"__threshold_0.98',
                'total__matrix_profile__feature_"75"__threshold_0.98',
                'total__mean_n_absolute_max__number_of_maxima_7',
                'percentil_05',
                'percentil_10',
                'percentil_25',
                'percentil_50',
                'percentil_75',
                'percentil_90',
                'percentil_95',
                'number_of_peak_2_above_average',
                'number_of_peak_over_3_above_average',
                'number_of_peak_over_5_above_average',
                'number_of_peak_over_10_above_average',
                'number_of_peak_over_20_above_average',
                'number_of_peak_over_30_above_average',
                'number_of_peak_over_50_above_average',
                'number_of_peak_over_100_above_average',
                'flag_always0',
                '1d_roll_sum',
                '2d_roll_sum',
                '5d_roll_sum',
                '7d_roll_sum',
                '14d_roll_sum',
                '30d_roll_sum',
                '60d_roll_sum',
                '90d_roll_sum',
                '1d_roll_median',
                '2d_roll_median',
                '5d_roll_median',
                '7d_roll_median',
                '14d_roll_median',
                '30d_roll_median',
                '60d_roll_median',
                '90d_roll_median',
                '1d_roll_mean',
                '2d_roll_mean',
                '5d_roll_mean',
                '7d_roll_mean',
                '14d_roll_mean',
                '30d_roll_mean',
                '60d_roll_mean',
                '90d_roll_mean',
                '5d_roll_std',
                '7d_roll_std',
                '14d_roll_std',
                '30d_roll_std',
                '60d_roll_std',
                '90d_roll_std',
                '1d_roll_max',
                '2d_roll_max',
                '5d_roll_max',
                '7d_roll_max',
                '14d_roll_max',
                '30d_roll_max',
                '60d_roll_max',
                '90d_roll_max',
                '1d_roll_min',
                '2d_roll_min',
                '5d_roll_min',
                '7d_roll_min',
                '14d_roll_min',
                '30d_roll_min',
                '60d_roll_min',
                '90d_roll_min',
                'shifted_1',
                'shifted_2',
                'shifted_7',
                'shifted_14',
                'running_min',
                'running_max']

    features_importance_short=[
        '1d_roll_sum',
        'shifted_diff_1',
        'shifted_diff_14',
        'shifted_diff_2',
        'shifted_diff_7',
        '5d_roll_std',
        '2d_roll_std',
        'change_2',
        'change_7',
        'change_1',
        'running_min',
        'total__fft_coefficient__attr_real__coeff_73',
        'total__fft_coefficient__attr_imag__coeff_88',
        'total__sum_values',
        'total_abs_energy',
        'ID',
        'date']
    
    features_importance_short_last=[
        '1d_roll_sum',
        '5d_roll_std',
        'shifted_diff_2',
        '2d_roll_std',
        'shifted_diff_1',
        'shifted_diff_14',
        'total__abs_energy',
        'shifted_diff_7',
        'change_7',
        'change_2'
        'total__sum_values',
        'total__fft_coefficient__attr_imag__coeff_46',
        'total__fft_coefficient__attr_real__coeff_10',
        'total__percentage_of_reoccurring_datapoints_to_all_datapoints',
        'total__time_reversal_asymmetry_statistic__lag_1',
        'ID',
        'date'
        ]
    
    return features_importance_short_last
