import pandas as pd
import numpy as np
import os
import time
import json
import gc
from tqdm import tqdm
from modules.config import config
from modules.compress import reduce_mem_usage


def gen_filename(stock_id, date_id, half_id):
    return f"snapshot_sym{stock_id}_date{date_id}_{half_id}.csv"


def combine_data(file_path: str = None) -> bool:
    '''将很多小的csv文件合并成一个大的文件'''

    file_path = file_path if file_path is not None else './AI量化模型预测挑战赛公开数据/train/'

    filenames = os.listdir(file_path)
    df_list = []
    file_list = []
    daily_sym_num_dict = dict()

    for date in range(64):
        daily_sym_num_dict[date] = 0
        for sym in range(10):
            am_name = gen_filename(sym, date, 'am')
            pm_name = gen_filename(sym, date, 'pm')
            if am_name in filenames and pm_name in filenames:
                # 上下午的数据都有
                daily_sym_num_dict[date] += 1
                file_list.append(am_name)
                file_list.append(pm_name)

    print('Loading Files ...')
    for name in tqdm(file_list):
        sub_df = pd.read_csv(file_path + name, index_col=0)
        df_list.append(sub_df)

    data = pd.concat(df_list, ignore_index=True)
    data.sort_values(by=['date', 'time', 'sym'],
                     inplace=True, ignore_index=True)

    # 将时间的串转化成整数
    # 这一步只出于压缩的需要，无需将时间转化成连续的整数数组
    # 因为在后续的pandas dataframe 转 numpy array 的过程中，就已经扔掉time这一个纬度了
    data['time'] = data['time'].apply(
        lambda x: int(time.mktime(time.strptime(x, '%H:%M:%S'))))
    data['time'] -= data.at[0, 'time']
    data['time'] += data['date'] * 10000

    data.to_pickle('./train_data.pickle')
    with open('daily_sym_num_dict.json', '+w') as f:
        json.dump(daily_sym_num_dict, f)

    return data


def df2array3(df: pd.DataFrame, feature_len: int, df_columns: list) -> np.ndarray:
    """
    将二维的dataframe转为三维的numpy数组
    数组的纬度分别为(total_sym, time, feature_size)
    例如(10, 3998, 2000)
    """

    df_pivot = df.pivot(index='time', columns=['sym'], values=df_columns)
    df_pivot = df_pivot[df_columns]
    array_3d = df_pivot.values.reshape(
        df_pivot.shape[0], len(df_pivot.columns.levels[1]), -1)
    array_3d = np.swapaxes(array_3d, 0, 1)
    # 在这种写法中，必须要保证dataframe 的最后五列是labels
    arr_features = array_3d[:, :, :feature_len]
    arr_labels = array_3d[:, :, feature_len:]

    # print(array_3d.shape, arr_features.shape,arr_labels.shape)

    return arr_features, arr_labels


def export_raw_data(df: pd.DataFrame, is_Train: bool = True,) -> bool:
    if is_Train is True:
        # write into npy file
        labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
        df_columns = df.columns.to_list()
        # 这里包含time,date,sym,feature1,feature2,...,featuren,label1,...,labeln

        for x in ['time', 'sym', 'date']:
            df_columns.remove(x)
        # df_columns 中包含了labels
        # 特征长度不包含labels，目的是把特征和标签切开
        feature_len = len(df_columns) - len(labels)

        # 我们也可以不管features的生成方法，单独保存labels。
        # 如此即使以后扩充features，也可以找到每一条观测与labels的一一对应

        train_data_list, train_label_list = [], []
        valid_data_list, valid_label_list = [], []  # 这是一种极其消耗内存的方法

        for date in range(config['train_days']):
            indexes = df[df['date'] == date].index  # 避免算两次
            arr_features, arr_labels = df2array3(df.loc[indexes, :],
                                                 feature_len, df_columns)
            train_data_list.append(arr_features)
            train_label_list.append(arr_labels)

        train_data = np.concatenate(train_data_list, axis=0)
        train_label = np.concatenate(train_label_list, axis=0)
        # (total_sym, time, feature_size)
        # 注意取labels时的顺序。0代表label_5，1代表label_10，以此类推
        np.save(f'./data/train_data.npy', train_data.astype(np.float32))
        np.save(f'./data/train_labels.npy', train_label.astype(np.int64))
        del train_data, train_data_list, train_label, train_label_list
        gc.collect()

        for date in range(config['train_days'], 64):
            indexes = df[df['date'] == date].index
            arr_features, arr_labels = df2array3(df.loc[indexes, :],
                                                 feature_len, df_columns)
            valid_data_list.append(arr_features)
            valid_label_list.append(arr_labels)

        valid_data = np.concatenate(valid_data_list, axis=0)
        valid_label = np.concatenate(valid_label_list, axis=0)
        np.save(f'./data/valid_data.npy', valid_data.astype(np.float32))
        np.save(f'./data/valid_labels.npy', valid_label.astype(np.int64))

        return True
    elif is_Train is False:
        # TODO: get test dataset
        pass

    return False


def export_data(data: pd.DataFrame, output_filename) -> bool:
    # write into npy file
    # 这里包含time,date,sym,feature1,feature2,...,featuren,label1,...,labeln
    data = reduce_mem_usage(data)
    data.sort_values(by=['date', 'time', 'sym'],
                     inplace=True, ignore_index=True)

    # 将时间的串转化成整数
    # 这一步只出于压缩的需要，无需将时间转化成连续的整数数组
    # 因为在后续的pandas dataframe 转 numpy array 的过程中，就已经扔掉time这一个纬度了
    data['time'] = data['time'].apply(
        lambda x: int(time.mktime(time.strptime(x, '%H:%M:%S'))))
    data['time'] -= data.at[0, 'time']
    data['time'] += data['date'] * 10000

    data.to_parquet(output_filename, engine='fastparquet')


def reshape_data(df: pd.DataFrame):

    df_columns = df.columns.to_list()
    for x in ['time', 'sym', 'date']:
        df_columns.remove(x)
    # df_columns 中包含了labels
    feature_len = len(df_columns)

    # 我们也可以不管features的生成方法，单独保存labels。
    # 如此即使以后扩充features，也可以找到每一条观测与labels的一一对应

    train_data_list = []
    valid_data_list = []  # 这是一种极其消耗内存的方法

    for date in range(config['train_days']):
        indexes = df[df['date'] == date].index  # 避免算两次
        arr_features, arr_labels = df2array3(df.loc[indexes, :],
                                             feature_len, df_columns)
        train_data_list.append(arr_features)

    train_data = np.concatenate(train_data_list, axis=0)
    # (total_sym, time, feature_size)
    # 注意取labels时的顺序。0代表label_5，1代表label_10，以此类推
    np.save(f'./data/train_data.npy', train_data.astype(np.float32))
    np.save(f'./data/train_labels.npy', train_label.astype(np.int64))
    del train_data, train_data_list, train_label, train_label_list
    gc.collect()

    for date in range(config['train_days'], 64):
        indexes = df[df['date'] == date].index
        arr_features, arr_labels = df2array3(df.loc[indexes, :],
                                             feature_len, df_columns)
        valid_data_list.append(arr_features)

    valid_data = np.concatenate(valid_data_list, axis=0)
    np.save(f'./data/valid_data.npy', valid_data.astype(np.float32))

    return True


if __name__ == '__main__':
    # data = pd.read_csv('input_sample.csv')
    # data = combine_data()
    # data = pd.read_pickle('./train_data.pickle')
    # export_raw_data(data)

    data = pd.read_parquet('./data/important_features.parquet',
                           engine='fastparquet',)  # no labels
    export_data(data, './data/compressed_important_features.parquet')
    print(data.info())
    print(data.head())
    # print('\n'.join(data.columns.to_list()))

    print('OK')
