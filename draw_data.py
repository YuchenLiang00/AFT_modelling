import pandas as pd
import os

def get_sample():
    data_path = input("set your data path (eg. '../../train'): ")
    data = pd.DataFrame()
    for file in os.listdir(data_path):
        if ("sym0" in file or "sym0" in file) and ("date0" in file or "date10" in file):
            csv_path = os.path.join(data_path, file)
            sub_data = pd.read_csv(csv_path, index_col=0)
            data = pd.concat([data, sub_data])
    data.reset_index(drop=True, inplace=True)
    data.to_csv('input_sample.csv', index=False)
    print('panel data has been saved in current folder: input_sample.csv')
    print(data)
    return data

if __name__ == '__main__':
    get_sample()
