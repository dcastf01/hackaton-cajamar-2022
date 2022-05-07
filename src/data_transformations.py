
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import Optional,List

logger=logging.getLogger(__name__)
class BasicDataTransformations:
    
    def __init__(self,
                columns_object:Optional[list]=None,
                columns_int:Optional[list]=None,
                columns_datetime:Optional[list]=None,
                table_name:Optional[str]="table_base",
                ):
        """_summary_

        Args:
            preprocess_to_do (List[dict]): example -->: {
                rename_column: {original_name_column: [holder_employee_number,last_update],
                }
            columns_object (Optional[list], optional): _description_. Defaults to None.
            columns_int (Optional[list], optional): _description_. Defaults to None.
            columns_datetime (Optional[list], optional): _description_. Defaults to None.
        """     
           
        self.columns_datetime= columns_datetime
        self.columns_object=columns_object
        self.columns_int=columns_int
        
        self.table_name=table_name

    def set_correct_format_and_useful_columns(self,df:pd.DataFrame):
        # logging.debug (f'columns with lowercase in {self.table_name}') 
        # df.columns= df.columns.str.lower()

        # 
        if self.columns_datetime:
            df=self.set_format_datetime_to_columns_in_df(df)
        if self.columns_object:
            df=self.set_format_object_to_columns_in_df(df)
        if self.columns_int:
            df=self.set_format_numeric_columns_in_df(df)
        return df
    def preprocess(self,df:pd.DataFrame):
        logging.debug (f'starting the preprocess in {self.table_name}')
        df=self.set_correct_format_and_useful_columns(df)
    
    def set_format_datetime_to_columns_in_df(self,df:pd.DataFrame,):
        for column in self.columns_datetime:
            df.loc[:,column]=pd.to_datetime(df[column],errors='raise',format='%Y-%m-%d')
        return df
    
    def set_format_object_to_columns_in_df(self,df:pd.DataFrame,):
        for column in self.columns_object:
            df.loc[:,column] = df[column].astype('object')
        return df

    def set_format_numeric_columns_in_df(self,df:pd.DataFrame,):
        for column in self.columns_int:
            try:
                df.loc[:,column] = pd.to_numeric(df[column])
            except ValueError as e:
                logging.error(f'error in set format numeric in some of this column \n {column}')
                logging.error(e)
                logging.warning(f'shape: \n {df.shape}')
                logging.warning(f'describe: \n {df.describe()}')
                
            except Exception as e :
                logging.error(f'error in set format numeric in the {column}')
                logging.error(df.columns.to_list())
                raise(e)
        return df