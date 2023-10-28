import pandas as pd
import numpy as np
from compress import reduce_mem_usage
import os
import time
from tqdm import tqdm


def combine_data(file_path: str = None) -> bool:
    '''将很多小的csv文件合并成一个大的文件'''
    if file_path is None:
        file_path = './AI量化模型预测挑战赛公开数据/train/'
    filenames = os.listdir(file_path)
    df_list = []
    for filename in tqdm(filenames):
        sub_df = pd.read_csv(file_path + filename, index_col=0)
        df_list.append(sub_df)

    data = pd.concat(df_list)

    return data


def compress_data(df) -> str:

    # modify the time series from str to float
    if df['time'].dtype == 'object':
        df['time'] = df['time'].apply(
            lambda x: time.mktime(time.strptime(x, '%H:%M:%S')))
        df['time'] -= df.at[0, 'time']

    # compress the df
    reduced_df = reduce_mem_usage(df)

    return reduced_df


def export_data(df: pd.DataFrame, is_Train: bool = True)->bool:
    if is_Train is True:
            
        # write into pickle file
        labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
        features = list(set(df.columns.to_list()) - set(labels))

        np.save('train_data.npy',df[features].to_numpy())
        np.save('train_labels.npy', df[labels].to_numpy())

        return True
    elif is_Train is False:
        # TODO: get test dataset
        pass
    
    return False
    
if __name__ == '__main__':
    data = pd.read_csv('input_sample.csv')
    export_data(compress_data(data))
    print('OK')
