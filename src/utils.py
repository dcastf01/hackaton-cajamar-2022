import multiprocessing
import pickle as pk
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

def transform_df_in_pivot_with_id_in_columns(df):

    pivot_by_dates=df.pivot(
    index='date',
    columns='ID',
    values='total'
    )
    return pivot_by_dates
def applyParallel_using_df_standar(df, func,desc:str='extractin in parallel somehting, next time add better desc',njobs=-1,**kwargs):
    if njobs==-1 :n_jobs=multiprocessing.cpu_count()  
    
    retLst = Parallel(n_jobs=njobs)(delayed(func)(df[df['ID']==id],id,**kwargs) for id in tqdm(df.ID.unique(),
                                                                                        desc=desc,
                                                                                        mininterval=25))
    return pd.concat(retLst)
def applyParallel_using_pivot_table(pivot_by_dates, func,desc:str='extractin in parallel somehting, next time add better desc',njobs=-1):
    if njobs==-1 :n_jobs=multiprocessing.cpu_count()  
    
    retLst = Parallel(n_jobs=njobs)(delayed(func)(pivot_by_dates[ts],ts) for ts in tqdm(pivot_by_dates.columns,
                                                                                        desc=desc,
                                                                                        mininterval=25))
    return pd.concat(retLst)
def save_model(pca,path):
    pk.dump(pca, open(path,"wb"))

def load_model(path):
    return pk.load(open(path,'rb'))


def extract_n_relevant_features(model,X_train,n=20):
    
    feats = {} # a dict to hold feature_name: feature_importance
    for feature, importance in zip(X_train.columns, model.feature_importances_):
        feats[feature] = importance #add the name/value pair 

    importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
    return importances.sort_values(by='Gini-importance',ascending=False).head(n)#.plot(kind='bar', rot=45)


def rmse(y_true,y_pred):
    error=mean_squared_error(y_true,y_pred,squared=False)
    return error

def get_range_of_dates(start_date:str='2020-02-01',periods:int=7,D_or_W='D'):
        start_date=start_date
        freq=D_or_W
        periods=periods
        dates = pd.date_range(
            start=start_date,
            periods=periods,  # An extra in case we include start
            freq=freq)
    
        return dates

def stan_init(m):
        """Retrieve parameters from a trained model.
        
        Retrieve parameters from a trained model in the format
        used to initialize a new Stan model.
        
        Parameters
        ----------
        m: A trained model of the Prophet class.
        
        Returns
        -------
        A Dictionary containing retrieved parameters of m.
        
        """
        res = {}
        for pname in ['k', 'm', 'sigma_obs']:
            res[pname] = m.params[pname][0][0]
        for pname in ['delta', 'beta']:
            res[pname] = m.params[pname][0]
        return res

def modify_path_if_week_periocity(path,D_or_W):

   
    path_split=os.path.split(path)
    root_path=path_split[0]
    name_file=path_split[-1]
    if D_or_W=='W':
        folder='week'
    elif D_or_W=='D':
        folder='days'
    new_folder=os.path.join(root_path,folder)
    os.makedirs(new_folder,exist_ok=True)
    path=os.path.join(new_folder,name_file)
    
    return path