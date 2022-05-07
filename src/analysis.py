from sklearn.decomposition import PCA, IncrementalPCA
import os
import matplotlib.pyplot as plt
import numpy as np 
def check_pca_in_df(df,extra_name:str='')->None:
    """remve id,date and total from the df before

    Args:
        df (_type_): without columns id, date or total
        extra_name (str, optional): _description_. Defaults to ''.
    """    
    def save_plot_explained_variance(name_path,pca):

        fig=plt.figure(figsize=(15,7))
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        fig.savefig(name_path,dpi=fig.dpi)
        plt.close()

    X=df.copy()#.drop(['ID','date','total'],axis=1)
    root_path=r'D:\programacion\Repositorios\datathon-cajamar-2022\data\08_reporting'
    # X=df_features_no_temporal.copy()
    # X.drop('total',inplace=True,axis=1)
    X.fillna(0,inplace=True)
    X.replace(np.inf,0,inplace=True)

    X.replace(-np.inf,0,inplace=True)
    n_components = 15
    ipca = IncrementalPCA(n_components=n_components, batch_size=256)
    X_ipca = ipca.fit_transform(X,)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    path_ipca=os.path.join(root_path,extra_name+'_ipca.png')
    save_plot_explained_variance(path_ipca,ipca)
    path_ipca=os.path.join(root_path,extra_name+'_pca.png')
    save_plot_explained_variance(path_ipca,ipca)
    
    