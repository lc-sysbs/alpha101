from scipy.stats import rankdata
from dateutil import parser
import numpy as np
import numpy.linalg as la  
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from WindPy import *  # 万矿数据接口
w.start()

## 内置函数
# 行业中性化
def neutral(data,ind):
    stocks = list(data.index)
    data_ = pd.DataFrame(ind.Data,index=['INDUSTRY'],columns=ind.Codes).T
    data_med = pd.get_dummies(data_,columns=['INDUSTRY'])
    X= np.array(data_med)
    y = data.values
    beta_ols = la.inv(X.T.dot(X)).dot(X.T).dot(y)  
    residual = y - X.dot(beta_ols)      
    return residual
def IndNeutralize(vwap,ind):
    vwap_ = vwap.fillna(value = 0)
    ind = w.wss(stocks, "industry_citic","unit=1;tradeDate="+ind.Times[0].strftime("%Y%m%d")+";industryType=1")
    for i in range(len(vwap_)):
        vwap_.iloc[i] = neutral(vwap_.iloc[i],ind)
    return vwap_
# 移动求和
def ts_sum(df, window):
    return df.rolling(window).sum()
# 移动平均
def sma(df, window):
    return df.rolling(window).mean()
# 移动标准差
def stddev(df, window):
    return df.rolling(window).std()
# 移动相关系数
def correlation(x, y, window):
    return x.rolling(window).corr(y)
# 移动协方差
def covariance(x, y, window):
    return x.rolling(window).cov(y)
# 在过去d天的时序排名
def rolling_rank(na):
    return rankdata(na)[-1]
def ts_rank(df,window):
    return df.rolling(window).apply(rolling_rank)
# 过去d天的时序乘积
def rolling_prod(na):
    return np.prod(na)
def product(df, window):
    return df.rolling(window).apply(rolling_prod)
# 过去d天最小值
def ts_min(df, window):
    return df.rolling(window).min()
# 过去d天最大值
def ts_max(df, window):
    return df.rolling(window).max()
# 当天取值减去d天前的值
def delta(df, period):
    return df.diff(period)
# d天前的值，滞后值
def delay(df, period):
    return df.shift(period)
# 截面数据排序，输出boolean值
def rank(df):
    return df.rank(pct=True,axis=1)
# 缩放时间序列，使其和为1
def scale(df, k=1):
    return df.mul(k).div(np.abs(df).sum())
# 过去d天最大值的位置
def ts_argmax(df, window):
    return df.rolling(window).apply(np.argmax) + 1 
# 过去d天最小值的位置
def ts_argmin(df, window):
    return df.rolling(window).apply(np.argmin) + 1 
# 线性衰减的移动平均加权
def decay_linear(df, period):
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df) # 生成与df大小相同的零数组
    na_lwma[:period, :] = df.iloc[:period, :] # 赋前period项的值
    na_series = df.as_matrix() 
    # 计算加权系数
    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # 从第period项开始计算数值
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma,index = df.index,columns = df.columns)

## 因子函数
def alpha1(close,returns):
    x = close
    x[returns < 0] = stddev(returns,20)
    alpha = rank(ts_argmax(x ** 2, 5))-0.5
    return alpha.fillna(value = 0)
def alpha2(Open,close,volume):
    r1 = rank(delta(np.log(volume), 2))
    r2 = rank((close - Open) / Open)
    alpha = -1 * correlation(r1,r2,6)
    return alpha.fillna(value = 0)
def alpha3(Open,volume):
    r1 = rank(Open)
    r2 = rank(volume)
    alpha = -1 * correlation(r1,r2,10)
    return alpha.replace([-np.inf, np.inf], 0).fillna(value = 0)
def alpha4(low):
    r = rank(low)
    alpha = -1 * ts_rank(r,9)
    return alpha.fillna(value = 0)
def alpha5(Open,vwap,close):
    alpha = (rank((Open - (ts_sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    return alpha.fillna(value = 0)
def alpha6(Open, volume):
    alpha = -1 * correlation(Open, volume, 10)
    return alpha.replace([-np.inf, np.inf], 0).fillna(value = 0)
def alpha7(volume,close):
    adv20 = sma(volume, 20)
    alpha = -1 * ts_rank(abs(delta(close, 7)), 60) * np.sign(delta(close, 7))
    alpha[adv20 >= volume] = -1
    return alpha.fillna(value = 0)
def alpha8(Open,returns):
    x1 = (ts_sum(Open, 5) * ts_sum(returns, 5))
    x2 = delay((ts_sum(Open, 5) * ts_sum(returns, 5)), 10)
    alpha = -1 * rank(x1-x2)
    return alpha.fillna(value = 0)
def alpha9(close):
    delta_close = delta(close, 1)
    x1 = ts_min(delta_close, 5) > 0
    x2 = ts_max(delta_close, 5) < 0
    alpha = -1 * delta_close
    alpha[x1 | x2] = delta_close
    return alpha.fillna(value = 0)
def alpha10(close):
    delta_close = delta(close, 1)
    x1 = ts_min(delta_close, 4) > 0
    x2 = ts_max(delta_close, 4) < 0
    x = -1 * delta_close
    x[x1 | x2] = delta_close
    alpha = rank(x)
    return alpha.fillna(value = 0)
def alpha11(vwap,close,volume):
    x1 = rank(ts_max((vwap - close), 3))
    x2 = rank(ts_min((vwap - close), 3))
    x3 = rank(delta(volume, 3))
    alpha = (x1 + x2) * x3
    return alpha.fillna(value = 0)
def alpha12(volume,close):
    alpha = np.sign(delta(volume, 1)) * (-1 * delta(close, 1))
    return alpha.fillna(value = 0)
def alpha13(volume,close):
    alpha = -1 * rank(covariance(rank(close), rank(volume), 5))
    return alpha.fillna(value = 0)
def alpha14(Open,volume,returns):
    x1 = correlation(Open, volume, 10).replace([-np.inf, np.inf], 0).fillna(value=0)
    x2 = -1 * rank(delta(returns, 3))
    alpha = x1 * x2
    return alpha.fillna(value = 0)
def alpha15(high,volume):
    x1 = correlation(rank(high), rank(volume), 3).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * ts_sum(rank(x1), 3)
    return alpha.fillna(value = 0)
def alpha16(high,volume):
    alpha = -1 * rank(covariance(rank(high), rank(volume), 5))
    return alpha.fillna(value = 0)
def alpha17(volume,close):
    adv20 = sma(volume, 20)
    x1 = rank(ts_rank(close, 10))
    x2 = rank(delta(delta(close, 1), 1))
    x3 = rank(ts_rank((volume / adv20), 5))
    alpha = -1 * (x1 * x2 * x3)
    return alpha.fillna(value = 0)
def alpha18(close,Open):
    x = correlation(close, Open, 10).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * (rank((stddev(abs((close - Open)), 5) + (close - Open)) + x))
    return alpha.fillna(value = 0)
def alpha19(close,returns):
    x1 = (-1 * np.sign((close - delay(close, 7)) + delta(close, 7)))
    x2 = (1 + rank(1 + ts_sum(returns, 250)))
    alpha = x1 * x2
    return alpha.fillna(value = 0)
def alpha20(Open,high,close,low):
    alpha = -1 * (rank(Open - delay(high, 1)) * rank(Open - delay(close, 1)) * rank(Open - delay(low, 1)))
    return alpha.fillna(value = 0)
def alpha21(volume,close):
    x1 = sma(close, 8) + stddev(close, 8) < sma(close, 2)
    x2 = sma(close, 8) - stddev(close, 8) > sma(close, 2)
    x3 = sma(volume, 20) / volume < 1
    alpha = pd.DataFrame(np.ones_like(close), index = close.index,columns = close.columns)
    alpha[x1 | x3] = -1 * alpha
    return alpha
def alpha22(high,volume,close):
    x = correlation(high, volume, 5).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * delta(x, 5) * rank(stddev(close, 20))
    return alpha.fillna(value = 0)
def alpha23(high,close):
    x = sma(high, 20) < high
    alpha = pd.DataFrame(np.zeros_like(close),index = close.index,columns = close.columns)
    alpha[x] = -1 * delta(high, 2).fillna(value = 0)
    return alpha
def alpha24(close):
    x = delta(sma(close, 100), 100) / delay(close, 100) <= 0.05
    alpha = -1 * delta(close, 3)
    alpha[x] = -1 * (close - ts_min(close, 100))
    return alpha.fillna(value = 0)
def alpha25(volume,returns,vwap,high,close):
    adv20 = sma(volume, 20)
    alpha = rank((((-1 * returns) * adv20) * vwap) * (high - close))
    return alpha.fillna(value = 0)
def alpha26(volume,high):
    x = correlation(ts_rank(volume, 5), ts_rank(high, 5), 5).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * ts_max(x, 3)
    return alpha.fillna(value = 0)
def alpha27(volume,vwap):
    alpha = rank((sma(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))
    alpha[alpha > 0.5] = -1
    alpha[alpha <= 0.5] = 1
    return alpha.fillna(value = 0)  
def alpha28(volume,high,low,close):
    adv20 = sma(volume, 20)
    x = correlation(adv20, low, 5).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = scale(((x + ((high + low) / 2)) - close))
    return alpha.fillna(value = 0)
def alpha29(close,returns):
    x1 = ts_min(rank(rank(scale(np.log(ts_sum(rank(rank(-1 * rank(delta((close - 1), 5)))), 2))))), 5)
    x2 = ts_rank(delay((-1 * returns), 6), 5)
    alpha = x1 + x2
    return alpha.fillna(value = 0) 
def alpha30(close,volume):
    delta_close = delta(close, 1)
    x = np.sign(delta_close) + np.sign(delay(delta_close, 1)) + np.sign(delay(delta_close, 2))
    alpha = ((1.0 - rank(x)) * ts_sum(volume, 5)) / ts_sum(volume, 20)
    return alpha.fillna(value = 0)
def alpha31(close,low,volume):
    adv20 = sma(volume,20)
    x1 = rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10))))
    x2 = rank((-1 * delta(close, 3)))
    x3 = np.sign(scale(correlation(adv20, low, 12).replace([-np.inf, np.inf], 0).fillna(value=0)))
    alpha = x1 + x2 + x3
    return alpha.fillna(value = 0)
def alpha32(close,vwap):
    x = correlation(vwap, delay(close, 5),230).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = scale(((sma(close, 7)) - close)) + 20 * scale(x)
    return alpha.fillna(value = 0)
def alpha33(Open,close):
    alpha = rank(-1 + (Open / close))
    return alpha
def alpha34(close,returns):
    x = (stddev(returns, 2) / stddev(returns, 5)).fillna(value = 0)
    alpha = rank(2 - rank(x) - rank(delta(close, 1)))
    return alpha.fillna(value = 0)
def alpha35(volume,close,high,low,returns):
    x1 = ts_rank(volume, 32)
    x2 = 1 - ts_rank(close + high - low, 16)
    x3 = 1 - ts_rank(returns, 32)
    alpha = (x1 * x2 * x3).fillna(value = 0)
    return alpha
def alpha36(Open,close,volume,returns,vwap):
    adv20 = sma(volume, 20)
    x1 = 2.21 * rank(correlation((close - Open), delay(volume, 1), 15))
    x2 = 0.7 * rank((Open- close))
    x3 = 0.73 * rank(ts_rank(delay((-1 * returns), 6), 5))
    x4 = rank(abs(correlation(vwap,adv20, 6)))
    x5 = 0.6 * rank((sma(close, 200) - Open) * (close - Open))
    alpha = x1 + x2 + x3 + x4 + x5
    return alpha.fillna(value = 0)
def alpha37(Open,close):
    alpha = rank(correlation(delay(Open - close, 1), close, 200)) + rank(Open - close)
    return alpha.fillna(value = 0)
def alpha38(close,Open):
    x = (close / Open).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * rank(ts_rank(Open, 10)) * rank(x)
    return alpha.fillna(value = 0)
def alpha39(volume,close,returns):
    adv20 = sma(volume, 20)
    x = -1 * rank(delta(close, 7)) * (1 - rank(decay_linear((volume / adv20), 9)))
    alpha = x *(1 + rank(ts_sum(returns, 250)))
    return alpha.fillna(value = 0)
def alpha40(high,volume):
    alpha = -1 * rank(stddev(high, 10)) * correlation(high, volume, 10)
    return alpha.fillna(value = 0)
def alpha41(high,low,vwap):
    alpha = pow((high * low),0.5) - vwap
    return alpha
def alpha42(vwap,close):
    alpha = rank((vwap - close)) / rank((vwap + close))
    return alpha
def alpha43(volume,close):
    adv20 = sma(volume, 20)
    alpha = ts_rank(volume / adv20, 20) * ts_rank((-1 * delta(close, 7)), 8)
    return alpha.fillna(value = 0)
def alpha44(high,volume):
    alpha = -1 *correlation(high, rank(volume), 5).replace([-np.inf, np.inf], 0).fillna(value=0)
    return alpha
def alpha45(close,volume):
    x = correlation(close, volume, 2).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = -1 * (rank(sma(delay(close, 5), 20)) * x * rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2)))
    return alpha.fillna(value = 0)
def alpha46(close):
    x = ((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)
    alpha = (-1 * (close - delay(close, 1)))
    alpha[x < 0] = 1
    alpha[x > 0.25] = -1
    return alpha.fillna(value = 0)
def alpha47(volume,close,high,vwap):
    adv20 = sma(volume, 20)
    alpha = ((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / sma(high, 5)) - rank((vwap - delay(vwap, 5)))
    return alpha.fillna(value = 0)
def alpha49(close):
    x = (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))
    alpha = (-1 * delta(close,1))
    alpha[x < -0.1] = 1
    return alpha.fillna(value = 0)
def alpha50(volume,vwap):
    alpha = -1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)
    return alpha.fillna(value = 0)
def alpha51(close):
    inner = (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))
    alpha = (-1 * delta(close,1))
    alpha[inner < -0.05] = 1
    return alpha.fillna(value = 0)
def alpha52(returns,volume,low):
    x = rank(((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220))
    alpha = -1 * delta(ts_min(low, 5), 5) * x * ts_rank(volume, 5)
    return alpha.fillna(value = 0)
def alpha53(close,high,low):
    alpha = -1 * delta((((close - low) - (high - close)) / (close - low).replace(0, 0.0001)), 9)
    return alpha.fillna(value = 0)
def alpha54(Open,close,high,low):
    x = (low - high).replace(0, -0.0001)
    alpha = -1 * (low - close) * (Open ** 5) / (x * (close ** 5))
    return alpha
def alpha55(high,low,close,volume):
    x = (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)).replace(0, 0.0001)
    alpha = -1 * correlation(rank(x), rank(volume), 6).replace([-np.inf, np.inf], 0).fillna(value=0)
    return alpha
def alpha56(returns,cap):
    alpha = 0 - (1 * (rank((sma(returns, 10) / sma(sma(returns, 2), 3))) * rank((returns * cap))))
    return alpha.fillna(value = 0)
def alpha57(close,vwap):
    alpha = 0 - 1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))
    return alpha.fillna(value = 0)
def alpha60(close,high,low,volume):
    x = ((close - low) - (high - close)) * volume / (high - low).replace(0, 0.0001)
    alpha = - ((2 * scale(rank(x))) - scale(rank(ts_argmax(close, 10))))
    return alpha.fillna(value = 0)
def alpha61(volume,vwap):
    adv180 = sma(volume, 180)
    alpha = rank((vwap - ts_min(vwap, 16))) < rank(correlation(vwap, adv180, 18))
    return alpha
def alpha62(volume,high,low,Open,vwap):
    adv20 = sma(volume, 20)
    x1 = rank(correlation(vwap, ts_sum(adv20, 22), 10))
    x2 =  rank(((rank(Open) + rank(Open)) < (rank(((high + low) / 2)) + rank(high))))
    alpha = x1 < x2
    return alpha*-1
def alpha64(high,low,Open,volume,vwap):
    adv120 = sma(volume, 120)
    x1 = rank(correlation(ts_sum(((Open * 0.178404) + (low * (1 - 0.178404))), 13),ts_sum(adv120, 13), 17))
    x2 = rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))
    alpha = x1 < x2
    return alpha*-1
def alpha65(volume,vwap,Open):
    adv60 = sma(volume, 60)
    x1 = rank(correlation(((Open * 0.00817205) + (vwap * (1 - 0.00817205))), ts_sum(adv60,9), 6))
    x2 = rank((Open - ts_min(Open, 14)))
    alpha = x1 < x2
    return alpha*-1
def alpha66(vwap,low,Open,high):
    x1 = rank(decay_linear(delta(vwap, 4), 7))
    x2 = (((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (Open - ((high + low) / 2))
    alpha = (x1 + ts_rank(decay_linear(x2, 11), 7)) * -1
    return alpha.fillna(value = 0)
def alpha68(volume,high,close,low):
    adv15 = sma(volume, 15)
    x1 = ts_rank(correlation(rank(high), rank(adv15), 9), 14)
    x2 = rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))
    alpha = x1 < x2
    return alpha*-1
def alpha71(volume,close,low,Open,vwap):
    adv180 = sma(volume, 180)
    x1 = ts_rank(decay_linear(correlation(ts_rank(close, 3), ts_rank(adv180,12), 18), 4), 16)
    x2 = ts_rank(decay_linear((rank(((low + Open) - (vwap + vwap))).pow(2)), 16), 4)
    alpha = x1
    alpha[x1 < x2] = x2
    return alpha.fillna(value = 0)
def alpha72(volume,high,low,vwap):
    adv40 = sma(volume, 40)
    x1 = rank(decay_linear(correlation(((high + low) / 2), adv40, 9), 10))
    x2 = rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(volume, 19), 7), 3))
    alpha = (x1 / x2.replace(0, 0.0001)).fillna(value = 0)
    return alpha
def alpha73(vwap,Open,low):
    x1 = rank(decay_linear(delta(vwap, 5), 3))
    x2 = delta(((Open * 0.147155) + (low * (1 - 0.147155))), 2) / ((Open *0.147155) + (low * (1 - 0.147155)))
    x3 = ts_rank(decay_linear((x2 * -1), 3), 17)
    alpha = x1
    alpha[x1 < x3] = x3
    return -1 * alpha.fillna(value = 0)
def alpha74(volume,close,high,vwap):
    adv30 = sma(volume, 30)
    x1 = rank(correlation(close, ts_sum(adv30, 37), 15))
    x2 = rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11))
    alpha = x1 < x2
    return alpha*-1
def alpha75(volume,vwap,low):
    adv50 = sma(volume, 50)
    alpha = rank(correlation(vwap, volume, 4)) < rank(correlation(rank(low), rank(adv50), 12))
    return alpha
def alpha77(volume,high,low,vwap):
    adv40 = sma(volume, 40)
    x1 = rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20))
    x2 = rank(decay_linear(correlation(((high + low) / 2), adv40, 3), 6))
    alpha = x1
    alpha[x1 > x2] = x2
    return alpha.fillna(value = 0)  
def alpha78(volume,low,vwap):
    adv40 = sma(volume, 40)
    x1 = rank(correlation(ts_sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 20), ts_sum(adv40, 20),7))
    x2 = rank(correlation(rank(vwap), rank(volume), 6))
    alpha = x1.pow(x2)
    return alpha.fillna(value = 0)  
def alpha81(volume,vwap):
    adv10 = sma(volume, 10)
    x1 = rank(np.log(product(rank((rank(correlation(vwap, ts_sum(adv10, 50), 8)).pow(4))), 15)))
    x2 = rank(correlation(rank(vwap), rank(volume), 5))
    alpha = x1 < x2
    return alpha*-1
def alpha83(high,low,close,volume):
    x = rank(delay(((high - low) / (ts_sum(close, 5) / 5)), 2)) * rank(rank(volume))
    alpha = x / (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close))
    return alpha.fillna(value = 0) 
def alpha84(vwap,close):
    alpha = pow(ts_rank((vwap - ts_max(vwap, 15)), 21), delta(close,5))
    return alpha.fillna(value = 0)
def alpha85(volume,high,close,low):
    adv30 = sma(volume, 30)
    x1 = rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,10))
    alpha = x1.pow(rank(correlation(ts_rank(((high + low) / 2), 4), ts_rank(volume, 10), 7)))
    return alpha.fillna(value = 0)
def alpha86(volume,close,Open,vwap):
    adv20 = sma(volume, 20)
    x1 = ts_rank(correlation(close, sma(adv20, 15), 6), 20)
    x2 = rank(((Open+ close) - (vwap + Open)))
    alpha = x1 < x2
    return alpha*-1
def alpha88(volume,Open,low,high,close):
    adv60 = sma(volume, 60)
    x1 = rank(decay_linear(((rank(Open) + rank(low)) - (rank(high) + rank(close))),8))
    x2 = ts_rank(decay_linear(correlation(ts_rank(close, 8), ts_rank(adv60,21), 8), 7), 3)
    alpha = x1
    alpha[x1 > x2] = x2
    return alpha.fillna(value = 0)
def alpha92(volume,high,low,close,Open):
    adv30 = sma(volume, 30)
    x1 = ts_rank(decay_linear(((((high + low) / 2) + close) < (low + Open)), 15), 19)
    x2 = ts_rank(decay_linear(correlation(rank(low), rank(adv30), 8), 7),7)
    alpha = x1
    alpha[x1 > x2] = x2
    return alpha.fillna(value = 0)
def alpha94(volume,vwap):
    adv60 = sma(volume, 60)
    x = rank((vwap - ts_min(vwap, 12)))
    alpha = x.pow(ts_rank(correlation(ts_rank(vwap,20), ts_rank(adv60, 4), 18), 3))* -1
    return alpha.fillna(value = 0)
def alpha95(volume,high,low,Open):
    adv40 = sma(volume, 40)
    x = ts_rank((rank(correlation(sma(((high + low)/ 2), 19), sma(adv40, 19), 13)).pow(5)), 12)
    alpha = rank((Open - ts_min(Open, 12))) < x
    return alpha.fillna(value = 0)
def alpha96(volume,vwap,close):
    adv60 = sma(volume, 60)
    x1 = ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 4),4), 8)
    x2 = ts_rank(decay_linear(ts_argmax(correlation(ts_rank(close, 7),ts_rank(adv60, 4), 4), 13), 14), 13)
    alpha = x1
    alpha[x1 < x2] = x2
    return alpha.fillna(value = 0)
def alpha98(volume,Open,vwap):
    adv5 = sma(volume, 5)
    adv15 = sma(volume, 15)
    x1 = rank(decay_linear(correlation(vwap, sma(adv5, 26), 5), 7))
    alpha = x1 - rank(decay_linear(ts_rank(ts_argmin(correlation(rank(Open), rank(adv15), 21), 9),7), 8))
    return alpha.fillna(value = 0)
def alpha99(volume,high,low):
    adv60 = sma(volume, 60)
    x1 = rank(correlation(ts_sum(((high + low) / 2), 20), ts_sum(adv60,20), 9))
    x2 = rank(correlation(low, volume, 6))
    alpha = x1 < x2
    return alpha*-1
def alpha101(close,Open,high,low):
    alpha = (close - Open) / ((high - low) + 0.001)
    return alpha

def alpha48(close,ind):
    r1 = (correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close
    r2 = ts_sum((pow((delta(close, 1) / delay(close, 1)),2)), 250)
    alpha = IndNeutralize(r1, ind) / r2
    return alpha.fillna(value = 0)
    
def alpha58(vwap,volume,ind):
    x = IndNeutralize(vwap, ind)
    alpha = -1 * ts_rank(decay_linear(correlation(x, volume, 4), 8), 6)
    return alpha.fillna(value = 0)

def alpha59(vwap,volume,ind):
    x = IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), ind)
    alpha = -1 * ts_rank(decay_linear(correlation(x, volume, 4), 16), 8)
    return alpha.fillna(value = 0)

def alpha63(volume,close,vwap,Open,ind):
    adv180 = sma(volume, 180).fillna(value = 0)
    r1 = rank(decay_linear(delta(IndNeutralize(close, ind), 2), 8))
    r2 = rank(decay_linear(correlation(((vwap * 0.318108) + (Open * (1 - 0.318108))), ts_sum(adv180, 37), 14), 12))
    alpha = -1 * (r1 - r2)
    return alpha.fillna(value = 0)
 
def alpha67(volume,vwap,high,ind):
    adv20 = sma(volume, 20)
    r = rank(correlation(IndNeutralize(vwap, ind), IndNeutralize(adv20, ind), 6))
    alpha = pow(rank(high - ts_min(high, 2)),r) * -1
    return alpha.fillna(value = 0)

def alpha69(volume,vwap,ind,close):
    adv20 = sma(volume, 20)
    r1 = rank(ts_max(delta(IndNeutralize(vwap, ind), 3), 5))
    r2 = ts_rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 5), 9)
    alpha = pow(r1,r2) * -1
    return alpha.fillna(value = 0)

def alpha70(close,ind,vwap):
    adv50 = sma(volume, 50).fillna(value = 0)
    r = ts_rank(correlation(IndNeutralize(close, ind), adv50, 18), 18)
    alpha = pow(rank(delta(vwap, 1)),r) * -1
    return alpha.fillna(value = 0)

def alpha76(volume,vwap,low,ind):
    adv81 = sma(volume, 81).fillna(value = 0)
    r1 = rank(decay_linear(delta(vwap, 1), 12))
    r2 = ts_rank(decay_linear(ts_rank(correlation(IndNeutralize(low, ind), adv81, 8), 20), 17), 19)
    alpha = r1
    alpha[r1 < r2] = r2
    return alpha.fillna(value = 0)

def alpha79(volume,close,Open,ind,vwap):
    adv150 = sma(volume, 150).fillna(value = 0)
    r1 = rank(delta(IndNeutralize(((close * 0.60733) + (Open * (1 - 0.60733))), ind), 1))
    r2 = rank(correlation(ts_rank(vwap, 4), ts_rank(adv150, 9), 15))
    alpha = (r1 < r2) * -1
    return alpha.fillna(value = 0)

def alpha80(Open,high,ind):
    adv10 = sma(volume, 10)
    r1 = rank(np.sign(delta(IndNeutralize(((Open * 0.868128) + (high * (1 - 0.868128))), ind), 4)))
    r2 = ts_rank(correlation(high, adv10, 5), 6)
    alpha = pow(r1,r2) * -1
    return alpha.fillna(value = 0)

def alpha82(Open,volume,ind):
    r1 = rank(decay_linear(delta(Open, 1), 15))
    r2 = ts_rank(decay_linear(correlation(IndNeutralize(volume, ind), ((Open * 0.634196) + (Open * (1 - 0.634196))), 17), 7), 13)
    alpha = r1
    alpha[r1 > r2] = r2
    return -1 * alpha.fillna(value = 0)

def alpha87(volume,close,vwap):
    adv81 = sma(volume, 81).fillna(value = 0)
    r1 = rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 2), 3))
    r2 = ts_rank(decay_linear(abs(correlation(IndNeutralize(adv81, ind), close, 13)), 5), 14)
    alpha = r1
    alpha[r1 < r2] = r2
    return -1 * alpha.fillna(value = 0)

def alpha89(low,vwap,ind):
    adv10 = sma(volume, 10)
    r1 = ts_rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 7), 6), 4)
    r2 = ts_rank(decay_linear(delta(IndNeutralize(vwap, ind), 3), 10), 15)
    alpha = r1 - r2
    return alpha.fillna(value = 0)

def alpha90(volume,close,ind,low):
    adv40 = sma(volume, 40).fillna(value = 0)
    r1 = rank((close - ts_max(close, 5)))
    r2 = ts_rank(correlation(IndNeutralize(adv40, ind), low, 5), 3)
    alpha = pow(r1,r2) * -1
    return alpha.fillna(value = 0)

def alpha91(close,ind,volume,vwap):
    adv30 = sma(volume, 30)
    r1 = ts_rank(decay_linear(decay_linear(correlation(IndNeutralize(close, ind), volume, 10), 16), 4), 5)
    r2 = rank(decay_linear(correlation(vwap, adv30, 4), 3))
    alpha = (r1 - r2) * -1
    return alpha.fillna(value = 0)

def alpha93(vwap,ind,volume,close):
    adv81 = sma(volume, 81).fillna(value = 0)
    r1 = ts_rank(decay_linear(correlation(IndNeutralize(vwap, ind), adv81, 17), 20), 8)
    r2 = rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 3), 16))
    alpha = r1 / r2
    return alpha.fillna(value = 0)

def alpha97(volume,low,vwap,ind):
    adv60 = sma(volume, 60).fillna(value = 0)
    r1 = rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), ind), 3), 20))
    r2 = ts_rank(decay_linear(ts_rank(correlation(ts_rank(low, 8), ts_rank(adv60, 17), 5), 19), 16), 7)
    alpha = (r1 - r2) * -1
    return alpha.fillna(value = 0)

def alpha100(volume,close,low,high,ind):
    adv20 = sma(volume, 20)
    r1 = IndNeutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), ind)
    r2 = 1.5 * scale(IndNeutralize(r1, ind))
    r3 = scale(IndNeutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), ind))
    alpha = -1 * (r2 - r3) * (volume / adv20)
    return alpha.fillna(value = 0)
