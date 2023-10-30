import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm


def reduce_mem_usage(df) -> pd.DataFrame:
    """
    压缩内存函数,文件大小没变化,占用内存减小
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



def test_npz():
    df:pd.DataFrame = pd.read_pickle('export_data.pickle')
    array = df.to_numpy()
    np.savez('export_data.npz',array)

if __name__ == '__main__':
    test_npz()
    # load_data()
    # parquet
    # npz 文件大小
    # test_npz()