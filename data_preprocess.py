import pandas as pd
import numpy as np
import os
import time
import gc
from tqdm import tqdm
from modules.config import config
from modules.compress import reduce_mem_usage


def load_all_data(file_path: str = None, filter: bool = False) -> bool:
    '''将很多小的csv文件合并成一个大的文件'''

    file_path = file_path if file_path is not None else './AI量化模型预测挑战赛公开数据/train/'
    file_names = os.listdir(file_path)
    if filter is True:
        file_list, _ = filter_all_day_symbols(file_names)
    else:
        file_list = file_names

    print('Loading Files ...')
    df_list = []
    for name in tqdm(file_list):
        sub_df = pd.read_csv(file_path + name, index_col=0)
        df_list.append(sub_df)

    data = pd.concat(df_list, ignore_index=True)
    data.sort_values(by=['date', 'time', 'sym'],
                     inplace=True, ignore_index=True)

    # 将时间的串转化成整数
    data['time'] = data['time'].apply(
        lambda x: int(time.mktime(time.strptime(x, '%H:%M:%S'))))
    data['time'] -= data.at[0, 'time']
    data = reduce_mem_usage(data)
    return data


def concat_data(features: pd.DataFrame, origin: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(features, origin, how='inner', on=['date', 'time', 'sym'])
    return df


def save_data_seperate(df: pd.DataFrame, output_path: str = './data/'):
    """ save data seperately """
    df[df['date'] < config['train_days']
                    ].to_parquet(output_path + 'train_data.parquet')
    df[df['date'] >= config['train_days']
                    ].to_parquet(output_path + 'valid_data.parquet')
    return


def gen_filename(stock_id, date_id, half_id):
    return f"snapshot_sym{stock_id}_date{date_id}_{half_id}.csv"


def filter_all_day_symbols(file_names: list) -> tuple[list, dict]:
    """get file names of which a symbol hase both am and pm trading data"""

    file_list = []
    daily_sym_num_dict = dict()

    for date in range(64):
        daily_sym_num_dict[date] = 0
        for sym in range(10):
            am_name = gen_filename(sym, date, 'am')
            pm_name = gen_filename(sym, date, 'pm')
            if am_name in file_names and pm_name in file_names:
                # 上下午的数据都有
                daily_sym_num_dict[date] += 1
                file_list.append(am_name)
                file_list.append(pm_name)

    return file_list, daily_sym_num_dict


def df2array3(df: pd.DataFrame, feature_len: int, df_columns: list) -> np.ndarray:
    """
    将二维的dataframe转为三维的numpy数组
    数组的纬度分别为(total_sym, time, feature_size)
    例如(10, 3998, 300)
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


def export_data(df: pd.DataFrame,index:pd.Index) -> bool:
    """ 将df转化为numpy数组并存储 """
    # df = reduce_mem_usage(df)
    # df.to_parquet('./data/compressed_all_data.parquet', engine='fastparquet')
    # gc.collect()
    df = df.loc[index, :]
    df_columns = df.columns.to_list()  # len: 434

    labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
    # 这里包含time,date,sym,feature1,feature2,...,featuren,label1,...,labeln

    for x in ['time', 'sym', 'date']:
        df_columns.remove(x)
    # df_columns 中包含了labels
    # 特征长度不包含labels，目的是把特征和标签切开
    feature_len = len(df_columns) - len(labels)  # len: 426

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

    for date in range(config['train_days'], 64):
        indexes = df[df['date'] == date].index
        arr_features, arr_labels = df2array3(df.loc[indexes, :],
                                                feature_len, df_columns)
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


def filter_index(index_df: pd.DataFrame) -> pd.DataFrame:
    """ 返回上下午都有值的股票的index """

    # print(index_df.shape)  # (2448775, 4)
    m1: np.ndarray = index_df[index_df['morning'] == 1].loc[:, ['date','sym']].values
    m0: np.ndarray = index_df[index_df['morning'] == 0].loc[:, ['date','sym']].values
    m1_set = set(map(tuple, m1))
    m0_set = set(map(tuple, m0))
    xor_set = m1_set ^ m0_set  # 存储了不同时在m1_set 和 m0_set中的(date, sym)元组
    # bool_series 中，False表示不应被去除的股票。标记为True，则表示应该被去掉
    bool_series = pd.Series(data=[False] * index_df.shape[0])

    for date, sym in xor_set:
        bool_series |= (index_df['date'] == date) & (index_df['sym'] == sym)
        # print(sum(bool_series == True))
    index_df = index_df[~bool_series]
    # print(index_df.shape)  # (2406796, 4)
    return index_df


def main():
    index: pd.DataFrame = pd.read_parquet(path='./data/all_data.parquet',
                           engine='fastparquet',
                           columns=['date', 'sym','time', 'morning'])
    filtered_index: pd.Index = filter_index(index).index
    del index
    gc.collect()

    data: pd.DataFrame = pd.read_parquet(path='./data/compressed_all_data.parquet',
                                         engine='fastparquet')

    export_data(data, filtered_index)
    print('OK')
    return True


def get_daily_sym_dict():
    index: pd.DataFrame = pd.read_parquet(path='./data/all_data.parquet',
                           engine='fastparquet',
                           columns=['date', 'sym','time', 'morning'])
    filtered_df: pd.DataFrame = filter_index(index)
    group_df = filtered_df.groupby(by='date')
    l = [len(sub_df['sym'].unique()) for (_, sub_df) in group_df]
    acc = np.array(l).cumsum()
    np.save('./data/cum_sym_num_dict.npy', acc)

    return True

if __name__ == '__main__':

    # main()
    get_daily_sym_dict()