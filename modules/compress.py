import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
    """
    压缩内存函数,文件大小没变化,占用内存减小
    """
    start_mem = df.memory_usage().sum() / 1024**2
    for col in tqdm(df.columns,'Compressing...'):
        col_type = df[col].dtypes
        if str(col_type)[:5] == 'float':
            # convert all float64 into float16
            df[col] = df[col].astype(np.float32)
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