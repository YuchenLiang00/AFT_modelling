import numpy as np
import pandas as pd
from copy import deepcopy
from modules.config import config
import torch
import json
from torch.utils.data import DataLoader, Dataset

""" dataset """

""" 
def __len__(self) -> int:
    # 返回的是训练集或验证集所有天的秒数和
    days = config['train_days
    '] if self.is_train is True else 64 - config['train_days']
    return self.data.shape[0] * days - config['seq_len'] # FIXME 要保证秒数指针是不能够跨过某一天的，并要重新探索临界条件。

def __getitem__(self, index) -> tuple[torch.tensor, torch.tensor]:
    cur_day, cur_sec = divmod(index, config['daily_secs']) # 获得现在index所指的天数和秒数
    cur_day_sym_num: int = self.daily_sym_num_dict[str(cur_day)] #获取今天有多少个股票

    present_sym_start_index: int = sum(list(self.daily_sym_num_dict.values())[:cur_day]) # 不包含currday
    # FIXME DataLoader要求每次传进去的tensor的shape是一样的。但是这样取很明显会不一样
    # Transformer真的关注时序信息吗？我们可不可以把所有的股票在时间上拼起来直接遍历？ ：可以
    X = self.data[present_sym_start_index:present_sym_start_index + cur_day_sym_num,
                    cur_sec,
                    :]
    y = self.labels[present_sym_start_index:present_sym_start_index + cur_day_sym_num, # 只取结束的秒
                    cur_sec, 
                    self.pred_label] # 这里的0表示以label_5为预测对象
    return X, y  """

def gen_dataiter(data, labels, daily_sym_num_dict, days,) -> tuple[np.ndarray, np.ndarray]:
    ''' DEPRECATED An iterator to get data sequentially'''
    # DEPRECATED 自定义的generator有内存泄露的风险
    for day in range(days):

        cur_day_sym_num: int = daily_sym_num_dict[str(day)]  # 获取今天有多少个股票
        present_sym_start_index: int = sum(
            list(daily_sym_num_dict.values())[:day])  # 不包含currday
        for sec in range(config['daily_secs'] - config['seq_len']):

            X = data[present_sym_start_index:present_sym_start_index + cur_day_sym_num,
                     sec:sec + config['seq_len'],
                     :]
            y = labels[present_sym_start_index:present_sym_start_index + cur_day_sym_num,
                       sec + config['seq_len'],
                       0]
            # Try to find out whether deepcopy helps the memory overuse.
            # Solved: NO.
            X, y = deepcopy(X), deepcopy(y)
            yield X, y


class DataIterable:
    def __init__(self,
                 data_dir: str,
                 label_dir: str,
                 dict_path: str = './daily_sym_num_dict.json',
                 is_train: bool = True,):
        '''
        DEPRECATED
        read data from file and return a Iterator
        reference: https://dogwealth.github.io/2021/07/08/Pytorch——DataLoader源码学习笔记/
        '''
        # DEPRECATED 自定义的generator有内存泄露的风险

        with open(dict_path, 'r') as f:
            self.daily_sym_num_dict: dict = json.load(f)  # 这里存储了每一天有多少只股票参与计算

        self.data: torch.tensor = torch.tensor(
            np.load(data_dir), dtype=torch.float32)
        self.labels: torch.tensor = torch.tensor(
            np.load(label_dir), dtype=torch.int64)
        # data (total_sym, time, feature_size)
        # 例如(48*10, 3998, 2000)
        self.days = config['train_days'] if is_train is True else 64 - \
            config['train_days']

    def __iter__(self):
        return gen_dataiter(self.data, self.labels, self.daily_sym_num_dict, self.days)


# Hints from GPT4
class LargeCSVDataset(Dataset):
    def __init__(self, csv_file, chunk_size=1000):
        self.csv_file = csv_file
        self.chunk_size = chunk_size

    def __len__(self):
        # This might not be the most efficient way to get the length
        # of a large file, but for the sake of the example:
        return sum(1 for _ in pd.read_csv(self.csv_file, chunksize=self.chunk_size))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Compute the start and end row of the chunk
        start_row = idx * self.chunk_size
        end_row = (idx + 1) * self.chunk_size

        chunk_data = pd.read_csv(self.csv_file, skiprows=range(1, start_row),
                                 nrows=self.chunk_size)

        # Process your data here (convert it to tensors, preprocess, etc.)
        # For this example, let's assume you are trying to predict a value
        # based on other features in the CSV.
        x = torch.tensor(chunk_data.drop(
            'target', axis=1).values, dtype=torch.float32)
        y = torch.tensor(chunk_data['target'].values, dtype=torch.float32)

        return x, y


'''
# Instantiate your dataset
dataset = LargeCSVDataset('path_to_large_file.csv', chunk_size=1000)

# Use DataLoader to handle batching
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Training loop
for epoch in range(0):
    for batch_x, batch_y in dataloader:
        # Your training code here
        pass
'''


# Ideas from DTQ
class RandomTimeDataset(Dataset):
    def init(self, data_val, data_idx, data_col, window_size=3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_val = data_val
        self.data_idx = data_idx.reset_index(drop=True)
        self.data_col = data_col
        self.window_size = window_size
        self.key_code_map = dict(enumerate(data_idx['sym'].unique()))

    def __len__(self):
        return len(self.key_code_map)

    def __getitem__(self, idx):
        code = self.key_code_map[idx]
        getIdx = self.data_idx.query("sym = @code")
        min_index, max_index = self.window_size, len(getIdx)
        if max_index < min_index:
            # 随机生成时间区域
            return self.__getitem__(np.random.randint(0, self.__len__()))

        end_index = np.random.randint(min_index, max_index)
        start_indx = end_index-self.window_size
        data = self.data_val[getIdx.index[start_indx:end_index], :]
        X, y = data[:, :-1], data[:, -1:]
        return torch.tensor(X), torch.tensor(y)


class HisData:

    def getRandomData(self):
        """ 
        返回时间顺序随机的batch，每次的batch_size固定，适用于Transformer模型
        return 
            X; torch[batch_size, seq_len, feature_size]
            Y: torch[batch_size, seq_len, output_size]
        """
        for _ in range(self.total_seq // self.seq_len):
            rdl = DataLoader(self.rtd, batch_size=self.batch_size)
            for data in rdl:
                yield data[0], data[1]


""" Transformer """

class DecoderLayer(nn.Module):

    def __init__(self, d_model, num_heads):
        super(DecoderLayer, self).__init__()
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads)  # TODO mask
        self.encoder_attention = nn.MultiheadAttention(d_model, num_heads)
        self.feedforward = nn.Sequential(nn.Linear(d_model, 4 * d_model),
                                         nn.ReLU(),
                                         nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, encoder_output):
        self_attention_output = self.self_attention(x, x, x)[0]
        self_attention_output = self.norm1(x + self_attention_output)

        encoder_attention_output = self.encoder_attention(
            self_attention_output, encoder_output, encoder_output)[0]
        encoder_attention_output = self.norm2(self_attention_output +
                                              encoder_attention_output)

        feedforward_output = self.feedforward(encoder_attention_output)
        output = self.norm3(encoder_attention_output + feedforward_output)
        return output


class Transformer(nn.Module):

    def __init__(self, config: dict):
        super(Transformer, self).__init__()
        self.config = config
        self.pe = PositionalEncoding(config['hidden_dim'], config['dropout'])
        self.input_layer = nn.Linear(config['input_dim'], config['hidden_dim'])

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(config['hidden_dim'], config['num_heads'])
            for _ in range(config['num_layers'])
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config['hidden_dim'], config['num_heads'])
            for _ in range(config['num_layers'])
        ])
        self.output_layer = nn.Linear(
            config['hidden_dim'], config['output_dim'])

    def forward(self, x):

        # Input layer
        x = self.feature_norm(x)
        x = self.input_layer(x)
        x = self.pe(x) if self.config['pos_enco'] is True else x

        # Encoder layers
        # encoder_output = x.transpose(0, 1)
        encoder_output = x
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        # Decoder layers
        # decoder_output = encoder_output[-1, :, :].unsqueeze(0)
        decoder_output = encoder_output
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output)

        # Output layer
        output = self.output_layer(decoder_output)
        return output


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = int(d_model / num_heads)

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        attention_weights = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(
            torch.tensor(self.depth, dtype=torch.float32))
        attention_weights = torch.softmax(attention_weights, dim=-1)

        output = torch.matmul(attention_weights, V)
        output = self._combine_heads(output)

        output = self.W_O(output)
        return output

    def _split_heads(self, tensor):
        tensor = tensor.view(tensor.size(0), -1, self.num_heads, self.depth)
        return tensor.transpose(1, 2)

    def _combine_heads(self, tensor):
        tensor = tensor.transpose(1, 2).contiguous()
        tensor = tensor.view(tensor.size(0), -1, self.num_heads * self.depth)
        return tensor


""" data_preprocess """


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
