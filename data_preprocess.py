from networkx import dfs_edges, to_numpy_array
import pandas as pd
import numpy as np
from modules.compress import reduce_mem_usage
import os
import time
from tqdm import tqdm


def gen_filename(stock_id, date_id, half_id):
    return f"snapshot_sym{stock_id}_date{date_id}_{half_id}.csv"


def combine_data(file_path: str = None) -> bool:
    '''将很多小的csv文件合并成一个大的文件'''

    file_path = file_path if file_path is not None else './AI量化模型预测挑战赛公开数据/train/'

    filenames = os.listdir(file_path)
    df_list = []
    file_list = []
    daily_sym_num = dict()

    for date in range(64):
        daily_sym_num[date] = 0
        for sym in range(10):
            am_name = gen_filename(sym, date, 'am')
            pm_name = gen_filename(sym, date, 'pm')
            if am_name in filenames and pm_name in filenames:
                # 上下午的数据都有
                daily_sym_num[date] += 1
                file_list.append(am_name)
                file_list.append(pm_name)

    print('Loading Files ...')
    for name in tqdm(file_list):
        sub_df = pd.read_csv(file_path + name, index_col=0)
        df_list.append(sub_df)

    data = pd.concat(df_list,ignore_index=True)
    data.sort_values(['date'],inplace=True, ignore_index=True)
    data['time'] = data['time'].apply(
            lambda x: int(time.mktime(time.strptime(x, '%H:%M:%S'))))
    data['time'] -= data.at[0, 'time']
    data.to_pickle('./train_data.pickle')
    return daily_sym_num


def compress_data(df, is_reduce: bool = False) -> str:

    # modify the time series from str to float
    if df['time'].dtype == 'object':
        df['time'] = df['time'].apply(
            lambda x: int(time.mktime(time.strptime(x, '%H:%M:%S'))))
        df['time'] -= df.at[0, 'time']

    # compress the df
    reduced_df = reduce_mem_usage(df) if is_reduce is True else df

    return reduced_df

def df2array3(df:pd.DataFrame, feature_len: int)->np.array:

    df_pivot = df.pivot(index='time', columns=['sym'], values=df.columns)
    array_3d = df_pivot.values.reshape(len(df_pivot), len(df_pivot.columns.levels[1]), -1)
    arr_features = array_3d[:,:,:feature_len]
    arr_labels = array_3d[:,:,feature_len:]
    # print(array_3d.shape, arr_features.shape,arr_labels.shape)

    return arr_features, arr_labels

def export_data(df: pd.DataFrame, is_Train: bool = True,)->bool:
    if is_Train is True:
        # write into npy file
        labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
        df_columns = df.columns.to_list()
        for x in labels+['date']:
            df_columns = df_columns.remove(x)
        features = df_columns
        
        # for date in tqdm(range(48)):
        #     indexes = df[df['date'] == date].index  # 避免算两次
        #     arr_features, arr_labels = df2array3(df.loc[indexes,:], len(features)) 
        #     # (time, sym, features+labels)
 
        #     np.save(f'./data/train_data_{date}.npy',arr_features.astype(np.float32))
        #     np.save(f'./data/train_labels_{date}.npy', arr_labels.astype(np.int64))

        indexes = df[df['date'] < 48].index  # 避免算两次
        arr_features, arr_labels = df2array3(df.loc[indexes,:].drop(columns=['date']), len(features)) 
        # (time, sym, features+labels)
        np.save(f'./data/train_data.npy', arr_features.astype(np.float32))
        np.save(f'./data/train_labels.npy', arr_labels.astype(np.int64))
        
        indexes = df[df['date'] >= 48].index
        arr_features, arr_labels = df2array3(df.loc[indexes,:].drop(columns=['date']), len(features)) 
        np.save(f'./data/valid_data.npy', arr_features.astype(np.float32))
        np.save(f'./data/valid_labels.npy', arr_labels.astype(np.int64))

        return True
    elif is_Train is False:
        # TODO: get test dataset
        pass

    return False

if __name__ == '__main__':
    # data = pd.read_csv('input_sample.csv')
    # data = combine_data()
    data = pd.read_pickle('./train_data.pickle')
    export_data(data)
    print('OK')
