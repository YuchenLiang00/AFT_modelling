import pandas as pd
import numpy as np
import os
import time
import gc
from tqdm import tqdm
from modules.config import config
from modules.compress import reduce_mem_usage


def df2array3(df: pd.DataFrame) -> np.ndarray:
    """
    将二维的dataframe转为三维的numpy数组
    数组的纬度分别为(total_sym, time, feature_size)
    例如(10, 3998, 300)
    """

    df_pivot = df.pivot(index='time', columns=['sym'])
    array_3d = df_pivot.values.reshape(
        df_pivot.shape[0], len(df_pivot.columns.levels[1]), -1)
    array_3d = np.swapaxes(array_3d, 0, 1)
    # print(array_3d.shape, arr_features.shape, arr_labels.shape)

    return array_3d


def export_data(df: pd.DataFrame,
                sym: pd.Series, morning: pd.Series) -> bool:
    """ 将df转化为numpy数组并存储 """
    # df = reduce_mem_usage(df)
    # df.to_parquet('./data/compressed_all_data.parquet', engine='fastparquet')
    # gc.collect()
    df.sort_values(by=['date', 'sym', 'time'],
                   ignore_index=True, inplace=True)
    df['sym'] = sym
    afternoon_stamp: int = df.loc[morning == 0, 'time'].iloc[0]

    df.loc[morning == 0, 'time'] -= afternoon_stamp
    feature_names = df.columns.to_list()  # len: 434
    # 这里包含time,date,sym,feature1,feature2,...,featuren,label1,...,labeln

    label_names = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']

    for x in ['time', 'sym', 'date'] + label_names:
        feature_names.remove(x)

    # 特征长度不包含labels，目的是把特征和标签切开 len: 426
    # 我们也可以不管features的生成方法，单独保存labels。
    # 如此即使以后扩充features，也可以找到每一条观测与labels的一一对应

    train_data_list, train_label_list = [], []
    valid_data_list, valid_label_list = [], []  # 这是一种极其消耗内存的方法

    for date in tqdm(range(config['train_days'])):
        indexes = (df['date'] == date)  # 避免算两次
        arr_features = df2array3(
            df.loc[indexes, ['time', 'sym'] + feature_names])
        arr_labels = df2array3(df.loc[indexes, ['time', 'sym'] + label_names])
        train_data_list.append(arr_features)
        train_label_list.append(arr_labels)
        gc.collect()

    train_data = np.concatenate(train_data_list, axis=0)
    train_label = np.concatenate(train_label_list, axis=0)
    # (total_sym, time, feature_size)
    # 注意取labels时的顺序。0代表label_5，1代表label_10，以此类推
    np.save(f'./data/train_data.npy', train_data.astype(np.float32))
    del train_data, train_data_list
    gc.collect()
    np.save(f'./data/train_labels.npy', train_label.astype(np.int64))
    del train_label, train_label_list
    gc.collect()

    for date in tqdm(range(config['train_days'], 64)):
        indexes = df[df['date'] == date].index
        arr_features = df2array3(
            df.loc[indexes, ['time', 'sym'] + feature_names])
        arr_labels = df2array3(df.loc[indexes, ['time', 'sym'] + label_names])
        valid_data_list.append(arr_features)
        valid_label_list.append(arr_labels)
        gc.collect()

    valid_data = np.concatenate(valid_data_list, axis=0)
    valid_label = np.concatenate(valid_label_list, axis=0)
    np.save(f'./data/valid_data.npy', valid_data.astype(np.float32))
    del valid_data, valid_data_list
    gc.collect()
    np.save(f'./data/valid_labels.npy', valid_label.astype(np.int64))
    del valid_label, valid_label_list
    gc.collect()

    return True


def combine_index(index_df: pd.DataFrame) -> pd.DataFrame:
    """ 合并sym和morning """
    index_df.sort_values(by=['date','sym','time'],
                         ignore_index=True, inplace=True)
    index_df['sym'] = index_df['sym'].astype('str')
    index_df['sym'] = index_df['sym'].str.cat(
        index_df['morning'].astype('str'), sep='_')
    return index_df['sym'], index_df['morning']


def main():
    index: pd.DataFrame = pd.read_parquet(path='./data/all_data.parquet',
                                          engine='fastparquet',
                                          columns=['date', 'sym', 'time', 'morning'])
    sym, morning = combine_index(index)
    del index
    gc.collect()
    data: pd.DataFrame = pd.read_parquet(path='./data/compressed_all_data.parquet',
                                         engine='fastparquet')
    export_data(data, sym, morning)
    print('OK')
    return True


if __name__ == '__main__':

    main()
