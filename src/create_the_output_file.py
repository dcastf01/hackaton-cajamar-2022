import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
from src.utils import applyParallel_using_df_standar
import warnings
warnings.filterwarnings("ignore")

def create_the_sum_between_dates(df,init_date:str,end_date:str):
    return df[(df['date']>=init_date) & (df['date']<=end_date)].Label.sum()

def create_the_two_rows_extra_per_id(df):
    df_aux=pd.DataFrame()
    df=df[['ID','Label','date']]
    init_date_1='2020-02-01' #cambiar por febrero
    end_date_1='2020-02-07'
    init_date_2='2020-02-08'
    end_date_2='2020-02-14'
    def process_to_get_the_semana(df_unique_id,id):
     
        df_per_id=df[df['ID']==id]
        df_per_id['Semana_1']=create_the_sum_between_dates(df_per_id,init_date_1,end_date_1)
        df_per_id['Semana_2']=create_the_sum_between_dates(df_per_id,init_date_2,end_date_2)
            
        df_per_id=df_per_id[['ID','Semana_1','Semana_2']].drop_duplicates()
        return df_per_id
    
    df_aux=applyParallel_using_df_standar   (df,process_to_get_the_semana,njobs=10)
    return df_aux

def create_pivot_table(df):
    df=df.pivot(
        index='ID',
        columns='date',
        values='Label'
    )

    return df


def create_output_df(df_daily,df_weekly):
    
    # values_per_week=create_the_two_rows_extra_per_id(df_daily)
    df_daily=df_daily[df_daily['date']<='2020-02-07']
    print(df_daily.date.unique())
    pivot_table_daily=create_pivot_table(df_daily)
    pivot_table_weekly=create_pivot_table(df_weekly)
    pivot_table_weekly.columns=['semana_1','semana_2']
    df=pd.merge(pivot_table_daily,pivot_table_weekly,on='ID')
    return df
    

def create_outputfile(df_daily,df_weekly,path_output='Cajamar_Universitat Politécnica de Valencia (UPV)_CanarIAs_1.txt'):
    #Sin cabecera ni nombres de filas.
    # Constará de 2.747 filas con 10 columnas cada fila:
    # • ID: ordenado de forma ascendente
    # • Dia_1: Predicción para el día 01/02/2020
    # • Dia_2: Predicción para el día 02/02/2020
    # • Dia_3: Predicción para el día 03/02/2020
    # • Dia_4: Predicción para el día 04/02/2020
    # • Dia_5: Predicción para el día 05/02/2020
    # • Dia_6: Predicción para el día 06/02/2020
    # • Dia_7: Predicción para el día 07/02/2020
    # • Semana_1: Predicción para la semana del 01/02 al 07/02/2020, ambos inclusive
    # • Semana_2: Predicción para la semana del 08/02 al 14/02/2020, ambos inclusive
    #Separando campos con “|”, el valor de la predicción en litros, y los decimales con “.” (incluir solo 2 decimales).
    # Ejemplo "Fichero respuesta" (primeras líneas)
    
    df_output=create_output_df(df_daily[['ID','Label','date']].copy(),df_weekly[['ID','Label','date']].copy())
    df_output.sort_values('ID',inplace=True,ascending=True)
    print(df_output.shape)
    print(df_output)

    df_output.to_csv(path_output,sep='|',index=True,header=False)