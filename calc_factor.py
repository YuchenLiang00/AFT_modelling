# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import multiprocessing
import gc

# 本地读取train数据集，list:[sym0,sym1,...,sym9]
def get_dataset(path):
    '''
    return: list
    '''
    #secIDs = [f"sym{i}" for i in range(10)]
    dataset = []
    for i in range(10):
        stock = 'sym' + str(i)
        data = pd.read_hdf(path, stock)
        dataset.append(data)
    return dataset

# TODO：按照calc_XXX命名因子计算函数，return DataFrame,注意对因子命名
# 因子计算函数

#%% By 秦宇航
def calc_SOIR(data: pd.DataFrame):
    '''
    Step Order Imbalance Ratio

    data: DataFrame 一支股票的数据

    return: DataFrame SOIR因子和SOIR_i序列, join date、time、sym、morning四列
    '''
    col1 = [f'n_bid{i}' for i in range(1,6)]
    col2 = [f'n_bsize{i}' for i in range(1,6)]
    col3 = [f'n_ask{i}' for i in range(1,6)]
    col4 = [f'n_asize{i}' for i in range(1,6)]
    time = data['time'].values

    n_bid = data[col1].values
    n_bsize = data[col2].values
    n_ask = data[col3].values
    n_asize = data[col4].values

    SOIR_it = (n_bsize - n_asize)/(n_bsize + n_asize)

    w = [1-(i-1)/5 for i in range(1,6)] 
    w = np.tile(w,(n_bid.shape[0],1))
    SOIR_t = (SOIR_it * w).sum(axis=1)

    res = np.concatenate([SOIR_it, SOIR_t.reshape(-1, 1)], axis=1)  
    res = pd.DataFrame(res, columns = [f'SOIR_{i}' for i in range(1,6)]+['SOIR'])
    
    return res

def calc_OI_i(data:pd.DataFrame):
    '''
    Order Imbalance

    data: DataFrame 一支股票的数据

    return: DataFrame SOIR因子和SOIR_i序列, join date、time、sym、morning四列
    '''
    col_b = [f'n_bsize{i}' for i in range(1,6)]
    col_a = [f'n_asize{i}' for i in range(1,6)]
    n_bsize = data[col_b].values
    n_asize = data[col_a].values

    res = pd.DataFrame(n_bsize - n_asize, columns = [f'OI_{i}' for i in range(1,6)])

    return res

def calc_AD(data:pd.DataFrame):
    '''
    Accumulated difference:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    col1 = [f'n_bid{i}' for i in range(1,6)]
    col2 = [f'n_bsize{i}' for i in range(1,6)]
    col3 = [f'n_ask{i}' for i in range(1,6)]
    col4 = [f'n_asize{i}' for i in range(1,6)]

    n_bid = data[col1].values
    n_bsize = data[col2].values
    n_ask = data[col3].values
    n_asize = data[col4].values

    factor_p = (n_ask - n_bid).sum(axis=1)
    factor_v = (n_asize - n_bsize).sum(axis=1)

    res = np.concatenate([factor_p.reshape(-1,1), factor_v.reshape(-1,1)], axis=1)
    res = pd.DataFrame(res, columns = ['AD_p', 'AD_v'])

    return res

def calc_PD(data:pd.DataFrame):
    '''
    Bid/Ask_price_distance:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    price_5 = data[['n_bid5','n_ask5']].values
    price_1 = data[['n_bid1','n_ask1']].values

    res = price_5 - price_1
    res = pd.DataFrame(res, columns = ['PD_Bid', 'PD_Ask'])

    return res

#%% By 梁朝越
def calc_PIR_i(data:pd.DataFrame):
    '''
    Price Imbalance Ratio:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    col1 = [f'n_bid{i}' for i in range(1,6)]
    col2 = [f'n_ask{i}' for i in range(1,6)]

    n_bid = data[col1].values
    n_ask = data[col2].values
    
    factor_i = (n_bid - n_ask)/np.where(n_bid + n_ask, n_bid + n_ask,np.nan)
    
    w = [1-(i-1)/5 for i in range(1,6)] 
    w = np.tile(w,(n_bid.shape[0],1))
    factor = (factor_i * w).sum(axis=1)

    res = np.concatenate([factor_i.reshape(-1,5), factor.reshape(-1,1)], axis=1)
    res = pd.DataFrame(res, columns = [f'PIR_{i}' for i in range(1,6)]+['PIR'])

    return res

def calc_CD_i(data:pd.DataFrame):
    '''
    Cumsum Difference:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    col1 = [f'n_bid{i}' for i in range(1,6)]
    col2 = [f'n_bsize{i}' for i in range(1,6)]
    col3 = [f'n_ask{i}' for i in range(1,6)]
    col4 = [f'n_asize{i}' for i in range(1,6)]

    n_bid = data[col1].values
    n_bsize = data[col2].values
    n_ask = data[col3].values
    n_asize = data[col4].values

    factor_p = (n_ask - n_bid).cumsum(axis=1)
    factor_v = (n_asize - n_bsize).cumsum(axis=1)

    res = np.concatenate([factor_p.reshape(-1,5), factor_v.reshape(-1,5)], axis=1)
    res = pd.DataFrame(res, columns = [f'CD_p_{i}' for i in range(1,6)]+[f'CD_v_{i}' for i in range(1,6)])

    return res

def calc_WAP_i(data:pd.DataFrame):
    '''
    Weighted Average Price:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    col1 = [f'n_bid{i}' for i in range(1,6)]
    col2 = [f'n_bsize{i}' for i in range(1,6)]
    col3 = [f'n_ask{i}' for i in range(1,6)]
    col4 = [f'n_asize{i}' for i in range(1,6)]

    n_bid = data[col1].values
    n_bsize = data[col2].values
    n_ask = data[col3].values
    n_asize = data[col4].values
    
    factor_i = (n_bid * n_asize + n_ask * n_bsize)/(n_asize + n_bsize)
    
    w = [1-(i-1)/5 for i in range(1,6)] 
    w = np.tile(w,(n_bid.shape[0],1))
    factor = (factor_i * w).sum(axis=1)

    res = np.concatenate([factor_i.reshape(-1,5), factor.reshape(-1,1)], axis=1)
    res = pd.DataFrame(res, columns = [f'WAP_{i}' for i in range(1,6)]+['WAP'])

    return res

def calc_MidP_i(data:pd.DataFrame):
    '''
    Midprice for 5 i:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    col1 = [f'n_bid{i}' for i in range(1,6)]
    col2 = [f'n_ask{i}' for i in range(1,6)]

    n_bid = data[col1].values
    n_ask = data[col2].values
    
    factor = (n_bid + n_ask)/2

    res = factor.reshape(-1,5)
    res = pd.DataFrame(res, columns = [f'MidP_{i}' for i in range(1,6)])

    return res

def calc_PD_1i(data:pd.DataFrame):
    '''
    Ask/Bid Price Distance from i=1:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    col1 = [f'n_bid{i}' for i in range(1,6)]
    col2 = [f'n_ask{i}' for i in range(1,6)]

    n_bid = data[col1].values
    n_ask = data[col2].values
    n_bid0 = np.tile(n_bid[:,0],(4,1)).T
    n_ask0 = np.tile(n_ask[:,0],(4,1)).T
    
    factor_b = -(n_bid[:,1:] - n_bid0)
    factor_a = (n_ask[:,1:] - n_ask0)

    res = np.concatenate([factor_b.reshape(-1,4), factor_a.reshape(-1,4)], axis=1)
    res = pd.DataFrame(res, columns = [f'PD_bid_1_{i}' for i in range(2,6)]+[f'PD_ask_1_{i}' for i in range(2,6)])

    return res

def calc_PD_diff(data:pd.DataFrame):
    '''
    Ask/Bid Price Distance from next/lasted i:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    col1 = [f'n_bid{i}' for i in range(1,6)]
    col2 = [f'n_ask{i}' for i in range(1,6)]

    n_bid = data[col1].values
    n_ask = data[col2].values
    
    factor_b = -np.diff(n_bid, axis=1)
    factor_a = np.diff(n_ask, axis=1)

    res = np.concatenate([factor_b.reshape(-1,4), factor_a.reshape(-1,4)], axis=1)
    res = pd.DataFrame(res, columns = [f'PD_diff_bid_{i}' for i in range(1,5)]+[f'PD_diff_ask_{i}' for i in range(1,5)])

    return res

def calc_VWAP_i(data:pd.DataFrame):
    '''
    Ask/Bid Weighted Average Price by Volume for each i(VWAP):
    Price Distance of VWAP for each i from i=1:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    col1 = [f'n_bid{i}' for i in range(1,6)]
    col2 = [f'n_bsize{i}' for i in range(1,6)]
    col3 = [f'n_ask{i}' for i in range(1,6)]
    col4 = [f'n_asize{i}' for i in range(1,6)]

    n_bid = data[col1].values
    n_bsize_0 = data[col2].values
    n_ask = data[col3].values
    n_asize_0 = data[col4].values
    
    n_bsize = np.where(n_bsize_0,n_bsize_0,np.nan)
    n_asize = np.where(n_asize_0,n_asize_0,np.nan)
    
    n_bid0 = np.tile(n_bid[:,0],(5,1)).T
    n_ask0 = np.tile(n_ask[:,0],(5,1)).T
    
    w = [1-(i-1)/5 for i in range(1,6)] 
    w = np.tile(w,(n_bid.shape[0],1))
    
    factor_b = (n_bid * n_bsize)/np.where(n_bsize,n_bsize,0)
    factor_a = (n_ask * n_asize)/np.where(n_asize,n_asize,0)
    factor_VWAP_bid = (factor_b * w).sum(axis=1)
    factor_VWAP_ask = (factor_a * w).sum(axis=1)
    
    factor_b1 = -(factor_b - n_bid0)
    factor_a1 = (factor_a - n_ask0)

    res = np.concatenate([factor_b.reshape(-1,5),factor_VWAP_bid.reshape(-1,1), factor_a.reshape(-1,5),factor_VWAP_ask.reshape(-1,1),
                          factor_b1.reshape(-1,5), factor_a1.reshape(-1,5)], axis=1)
    res = pd.DataFrame(res, columns = [f'VWAP_bid_{i}' for i in range(1,6)]+['VWAP_bid'] + [f'VWAP_ask_{i}' for i in range(1,6)]+['VWAP_ask'] + 
                                      [f'VWAP_diff_bid_{i}' for i in range(1,6)]+[f'VWAP_diff_ask_{i}' for i in range(1,6)])

    return res

#%% By 李航宇
def get_VOI(row):
    for i in range(1, 6):
        bid_diff = row[f'bid{i}_diff']
        bsize_diff = row[f'bsize{i}_diff']
        n_bsize = row[f'n_bsize{i}']
        ask_diff = row[f'ask{i}_diff']
        n_asize = row[f'n_asize{i}']
        asize_diff = row[f'asize{i}_diff']

        delta_bsize = np.where(bid_diff > 0, n_bsize, np.where(bid_diff < 0, 0, bsize_diff))
        delta_asize = np.where(ask_diff > 0, 0, np.where(ask_diff < 0, n_asize, asize_diff))

        row[f'VOI{i}'] = delta_bsize - delta_asize
    return row

def calc_VOI(data: pd.DataFrame):
    '''
        Volume Order Imbalance 量订单失衡因子
    '''

    res = data.copy()

    # 添加VOI各列，VOI1-VOI5
    for i in range(1, 6):
        res[f'VOI{i}'] = 0
    res['VOI'] = 0

    # 五档VOI计算所需数据
    for i in range(1, 6):
        # bid price时序差分，第一行默认置为0
        res[f'bid{i}_diff'] = res[f'n_bid{i}'].diff().fillna(0)
        # ask price时序差分，第一行默认置为0
        res[f'ask{i}_diff'] = res[f'n_ask{i}'].diff().fillna(0)
        # bid size时序差分，第一行默认置为第一个数
        res[f'bsize{i}_diff'] = res[f'n_bsize{i}'].diff().fillna(res[f'n_bsize{i}'].iloc[0])
        # ask size时序差分，第一行默认置为第一个数
        res[f'asize{i}_diff'] = res[f'n_asize{i}'].diff().fillna(res[f'n_asize{i}'].iloc[0])

    # 计算各档VOI
    res = res.apply(get_VOI, axis=1)
    # 加权VOI，权重比例为5-1
    for i in range(1, 6):
        res[f'VOI{i}'] = res[f'VOI{i}'] * (6 - i)
    res['VOI'] = res[[f'VOI{i}' for i in range(1, 6)]].sum(axis=1) / 15
    res_cols = [f'VOI{i}' for i in range(1, 6)] + ['VOI']

    return res[res_cols]

def get_OFI(row):
    for i in range(1, 6):

        bid_diff = row[f'bid{i}_diff']
        bsize_last = row[f'bsize{i}_last']
        bsize_diff = row[f'bsize{i}_diff']
        n_bsize = row[f'n_bsize{i}']
        ask_diff = row[f'ask{i}_diff']
        asize_last = row[f'asize{i}_last']
        asize_diff = row[f'asize{i}_diff']
        n_asize = row[f'n_asize{i}']

        delta_b_size = np.where(bid_diff > 0, n_bsize, np.where(bid_diff < 0, -bsize_last, bsize_diff))
        delta_a_size = np.where(ask_diff > 0, -asize_last, np.where(ask_diff < 0, n_asize, asize_diff))
        row[f'OFI{i}'] = delta_b_size - delta_a_size
    return row

def calc_OFI(data: pd.DataFrame):
    '''
        Order Flow Imbalance 订单失衡因子
    '''

    res = data.copy()

    # 添加VOI各列，OFI1-OFI5
    for i in range(1, 6):
        res[f'OFI{i}'] = 0
    res['OFI'] = 0

    # 五档VOI计算所需数据
    for i in range(1, 6):
        # bid price时序差分，第一行默认置为0
        res[f'bid{i}_diff'] = res[f'n_bid{i}'].diff().fillna(0)
        # ask price时序差分，第一行默认置为0
        res[f'ask{i}_diff'] = res[f'n_ask{i}'].diff().fillna(0)
        # bid size时序差分，第一行默认置为第一个数
        res[f'bsize{i}_diff'] = res[f'n_bsize{i}'].diff().fillna(res[f'n_bsize{i}'].iloc[0])
        # ask size时序差分，第一行默认置为第一个数
        res[f'asize{i}_diff'] = res[f'n_asize{i}'].diff().fillna(res[f'n_asize{i}'].iloc[0])
        # 上一时刻bid size
        res[f'bsize{i}_last'] = res[f'n_bsize{i}'].shift(1).fillna(0)
        # 上一时刻ask size
        res[f'asize{i}_last'] = res[f'n_asize{i}'].shift(1).fillna(0)

    # 计算各档OFI
    res = res.apply(get_OFI, axis=1)
    # 加权OFI，权重比例为1-5
    for i in range(1, 6):
        res[f'OFI{i}'] = res[f'OFI{i}'] * i
    res['OFI'] = res[[f'OFI{i}' for i in range(1, 6)]].sum(axis=1) / 15

    res_cols = [f'OFI{i}' for i in range(1, 6)] + ['OFI']
    return res[res_cols]

def calc_VOSC(data: pd.DataFrame):
    '''
        Volume Oscillator 成交量量振荡因子
    '''

    res = data.copy()

    # 5, 10, 20, 40, 60tick的移动平均成交量
    for i in [5, 10, 20, 40, 60]:
        res[f'VMA{i}'] = res['amount_delta'].rolling(i, min_periods=1).mean()

    # 1-5, 1-10, 1-20, 1-40, 1-60, 5-10, 5-20, 5-40, 5-60, 10-60的成交量差
    diff_list = [
        [1, 5], [1, 10], [1, 20], [1, 40], [1, 60],
        [5, 10], [5, 40], [5, 60],
        [10, 60]
    ]

    res.rename(columns={'amount_delta': 'VMA1'}, inplace=True)
    for i, j in diff_list:
        res[f'VOSC{i}-{j}'] = res[f'VMA{i}'] - res[f'VMA{j}']

    res_cols = [f'VOSC{i}-{j}' for i, j in diff_list]
    return res[res_cols]

#%% By 牛奕彤
def calc_OIR(data:pd.DataFrame):
    '''
    Order Imbalance Ratio

    data: DataFrame 一支股票的数据

    return: DataFrame OIR因子和OIR_i序列
    '''
    col2 = [f'n_bsize{i}' for i in range(1,6)]
    col4 = [f'n_asize{i}' for i in range(1,6)]

    n_bsize = data[col2].values
    n_asize = data[col4].values

    w = [1-(i-1)/5 for i in range(1,6)]
    weight_sum = sum(w)
    w = np.tile(w,(n_bsize.shape[0],1))
    wbsize = (n_bsize * w).sum(axis=1) / weight_sum
    wasize = (n_asize * w).sum(axis=1) / weight_sum
    OIR_it = (wbsize - wasize)/(wbsize + wasize)

    res = pd.DataFrame(OIR_it.reshape(-1, 1), columns = ['OIR'])

    return res

def calc_MPC(data:pd.DataFrame):
    '''
    Midpoint price change:
    
    data: DataFrame 一支股票的数据

    return: DataFrame MPC MPCmax MPCskew
    '''
    col1 = [f'n_bid{i}' for i in range(1,6)]
    col3 = [f'n_ask{i}' for i in range(1,6)]
    col_MPC = [f'MPC_{i}' for i in range(1,6)]

    n_bid = data[col1].values
    n_ask = data[col3].values

    mp = (n_bid + n_ask) / 2

    mp_diff = np.concatenate([np.full((1,5), np.nan), np.diff((n_bid + n_ask) / 2,axis=0)])
    mp_shift = np.concatenate([np.full((1,5), np.nan), mp[:-1]])
    mpc = mp_diff/np.where(mp_shift,mp_shift,np.nan)
    mpc = pd.DataFrame(mpc, columns = [f'MPC_{i}' for i in range(1,6)])

    mpc = pd.concat([data['date'],mpc], axis=1)

    gp = mpc.groupby('date')

    mpc_max = gp.max()
    mpc_max.columns = [f'MPC_max_{i}' for i in range(1,6)]
    mpc_skew = gp.skew()
    mpc_skew.columns = [f'MPC_skew_{i}' for i in range(1,6)]

    res = pd.concat([mpc_max, mpc_skew], axis=1).reset_index()
    res = pd.merge(mpc, res, how = 'outer',on='date')
    del res['date']

    return res

def calc_MAX(data:pd.DataFrame):
    '''
    cumprod of top 10% MAX  3s return:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    data['snapshotret'] = data['n_close'].diff() / (data['n_close'].shift(1) + 1) + 1
    data.loc[0, 'snapshotret'] = 1
    data = data.sort_values(by='snapshotret', ascending=False)
    MAX = data.groupby('date')['snapshotret'].apply(lambda x: x.head(int(len(x) / 10)).cumprod().tail(1))
    # MAX = pd.merge(MAX.reset_index(), data[['date', 'time']], on='date').sort_values(by=[['date', 'time']])
    # del MAX[['date', 'time']]
    
    MAX = pd.merge(MAX.reset_index()[['date','snapshotret']], data[['date']], on='date',how = 'right').sort_values(by=['date']).reset_index()  

    res = MAX[['snapshotret']]
    res.columns = ['MAX']
    return res

def calc_RSJ(data:pd.DataFrame):
    '''
    @nyt
    Relative Signed Jump:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    data['snapshotret'] = data['n_close'].diff() / data['n_close'].shift(1)
    # data['snapshotret'].loc[0] = 0
    data.loc[0, 'snapshotret'] = 1
    # yfc理解这两列就是一列吧
    data['positivejump'] = data['snapshotret'].apply(lambda x: 1 if x > 0 else 0)
    data['negativejump'] = data['snapshotret'].apply(lambda x: 1 if x < 0 else 0)
    # RVup = data.groupby('date')[['snapshotret', 'positivejump']].apply(lambda x: ((x['snapshotret']**2) * x['positivejump']).sum())
    # RVdown = data.groupby('date')[['snapshotret', 'negativejump']].apply(lambda x: ((x['snapshotret']**2) * x['negativejump']).sum())
    # RV = data.groupby('date')['snapshotret'].apply(lambda x: (x**2).sum())
    # RSJ = (RVup - RVdown) / RV
    
    data['RVup'] = data['snapshotret']**2 * data['positivejump']
    data['RVdown'] = data['snapshotret']**2 * data['negativejump']
    data['RV'] = data['snapshotret']**2

    RVup = data.groupby('date')['RVup'].sum()
    RVdown = data.groupby('date')['RVdown'].sum()
    RV = data.groupby('date')['RV'].sum()

    RSJ = (RVup - RVdown) / RV
    # print(RSJ.info())
    # RSJ = pd.merge(RSJ.reset_index(), data[['date', 'time']], on='date').sort_values(by=[['date', 'time']])
    RSJ = pd.merge(RSJ.reset_index(), data[['date']], on='date',how = 'right').sort_values(by=['date'])
    # del RSJ[['date', 'time']]
    # print(RSJ)
    res = RSJ.iloc[:,[1]].reset_index(drop=True)
    res.columns = ['RSJ']
    return res

#%% By 闫富晨

def calc_Patience(data:pd.DataFrame):
    '''
    @ yfc
    patience:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    slope_ask = np.nan_to_num((data['n_ask5'] - data['n_ask1'])/(data['n_asize2']+data['n_asize3']+data['n_asize4']+data['n_asize5']))
    slope_bid = np.nan_to_num((data['n_bid1'] - data['n_bid5'])/(data['n_bsize2']+data['n_bsize3']+data['n_bsize4']+data['n_bsize5']))
    slope_diff = slope_ask - slope_bid
    patience_all = (slope_ask - slope_bid)/(slope_ask + slope_bid)
    res = np.concatenate([slope_ask.reshape(-1,1), slope_bid.reshape(-1,1),slope_diff.reshape(-1,1), patience_all.reshape(-1,1)], axis=1)
    res = pd.DataFrame(res, columns = ['slope_ask','slope_bid','slope_diff','patience_all'])
    return res

def calc_Dispersion(data:pd.DataFrame):
    '''
    @ yfc
    disperation:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    lenth = 5
    numerator_bid = 0
    dominator_bid = 0
    numerator_ask = 0
    dominator_ask = 0
    for i in range(1,lenth):
        numerator_bid += data[f'n_bsize{i}'] * (data[f'n_bid{i}'] - data[f'n_bid{i+1}'])
        numerator_ask += data[f'n_asize{i}'] * (data[f'n_ask{i}'] - data[f'n_ask{i+1}'])
        dominator_bid += data[f'n_bsize{i}']
        dominator_ask += data[f'n_asize{i}']
    
    disp = (numerator_bid/dominator_bid + numerator_ask/dominator_ask)/2
    disp =  np.nan_to_num(disp)
    res = pd.DataFrame(disp.reshape(-1, 1), columns = ['Dispersion'])
    return res

def calc_AVE_Slope(data:pd.DataFrame):
    '''
    @ yfc
    average slope:
    
    data: DataFrame 一支股票的数据

    return: DataFrame
    '''
    lenth = 5
    numeratorbid = 0
    dominatorbid = 0
    numeratorask = 0
    dominatorask = 0
    for i in range(1, lenth):
        numeratorbid += data[f'n_bsize{i}']/data[f'n_bsize{i+1}'] - 1
        dominatorbid += data[f'n_bid{i}']/data[f'n_bid{i+1}'] - 1
        numeratorask += data[f'n_asize{i+1}']/data[f'n_asize{i}'] - 1
        dominatorask += data[f'n_ask{i+1}']/data[f'n_ask{i}'] - 1
        # print(data[f'n_ask{i}'])
    aveslope = (numeratorbid/(dominatorbid+1e-10) + numeratorask/(dominatorask+1e-10))/2/(lenth-1)
    aveslope =  np.nan_to_num(aveslope)
    res = pd.DataFrame(aveslope.reshape(-1, 1), columns = ['AVEslope'])
    return res

#%% 
# 因子计算
def calc_factor(data:pd.DataFrame):
    '''
    计算所有因子
    data: DataFrame 一支股票的数据

    return: DataFrame of factor 
    '''   
    
    # TODO: 写完新的计算因子函数把方法名加在这里
    methods = ['SOIR','PD','AD',\
               'OI_i','PIR_i', 'CD_i', 'WAP_i', 'MidP_i', 'PD_1i','PD_diff', 'VWAP_i',\
               'VOI', 'OFI', 'VOSC',\
               'OIR','MPC', 'MAX', 'RSJ',
               'Dispersion','AVE_Slope','Patience'] 

    # factor = pd.concat([eval(f'calc_{m}(data)') for m in methods], axis=1)
    factor = pd.DataFrame()
    for m in methods:
        f = eval(f'calc_{m}(data)')
        factor = pd.concat([factor, f], axis=1)
        del f
        gc.collect()

    factor = pd.concat([data.loc[:,['date','time','sym','morning']], factor], axis=1)

    return factor



if __name__ == '__main__':

    path = './data/merged_data/train_data.h5'
    dataset = get_dataset(path)

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes = num_processes)
    res = pool.map(calc_factor, dataset)
    pool.close()
    pool.join() 
    
    df_factor = pd.concat(res)
    
    print(df_factor.head())
    df_factor.to_parquet('./factor.parquet')
    # h5file = pd.HDFStore('data/factor.h5', mode='w')
    # h5file['factor'] = df_factor
    # h5file.close()


    