import pandas as pd
import numpy as np
import time
import os

from tqdm import tqdm


def reduce_mem_usage(df) -> pd.DataFrame:
    """
    压缩内存函数,文件大小没变化,占用内存减小
    TODO: 还可以进一步压缩将所有的float64压缩成16
    """
    df = df.copy()
    start_mem = df.memory_usage().sum() / 1024**2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(
                        np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(
                        np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(
                        np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(
                        np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # convert all float64 into float16
                df[col] = df[col].astype(np.float32)
            # else:
            #     if c_min > np.finfo(np.float16).min and c_max < np.finfo(
            #             np.float16).max:
            #         df[col] = df[col].astype(np.float16)
            #     elif c_min > np.finfo(np.float32).min and c_max < np.finfo(
            #             np.float32).max:
            #         df[col] = df[col].astype(np.float32)
            #     else:
            #         df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage before optimization is {round(start_mem,3)} MB")
    print(f"Memory usage after optimization is {round(end_mem,3)} MB")
    print(f"after / before = {end_mem / start_mem:.2%}")
    return df


def compress_data(filepath: str = None) -> str:

    if filepath is None:
        filepath = 'input_sample.csv'
        df = pd.read_csv(filepath)
    elif filepath.endswith('.pickle') or filepath.endswith('.pkl'):
        df = pd.read_pickle(filepath)
    else:
        df = pd.read_csv(filepath)

    # modify the time series from str to float
    if df['time'].dtype == 'object':
        df['time'] = df['time'].apply(
            lambda x: time.mktime(time.strptime(x, '%H:%M:%S')))
        df['time'] -= df.at[0, 'time']

    # compress the df
    reduced_df = reduce_mem_usage(df)
    # write into pickle file
    labels = ['label_5', 'label_10', 'label_20', 'label_40', 'label_60']
    features = list(set(reduced_df.columns.to_list()) - set(labels))
    df_label = reduced_df[labels]
    data = reduced_df[features]

    data.to_pickle('compressed_data.pickle')
    df_label.to_pickle('compressed_labels.pickle')

    return reduced_df


def load_data(file_path: str = None, output_name: str = None) -> bool:
    if file_path is None:
        file_path = './AI量化模型预测挑战赛公开数据/train/'
    filenames = os.listdir(file_path)
    df_list = []
    for filename in tqdm(filenames):
        sub_df = pd.read_csv(file_path + filename, index_col=0)
        df_list.append(sub_df)

    data = pd.concat(df_list)
    print(data)
    output_name = output_name if output_name is not None else 'export_data.pickle'
    data.to_pickle(output_name)

    return data

def test_npz():
    df:pd.DataFrame = pd.read_pickle('export_data.pickle')
    array = df.to_numpy()
    np.savez('export_data.npz',array)

if __name__ == '__main__':
    compress_data()
    # load_data()
    # parquet
    # npz 文件大小
    # test_npz()