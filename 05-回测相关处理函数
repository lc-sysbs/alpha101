from WindPy import *
import numpy as np
import pandas as pd
import numpy.linalg as la
import seaborn as sns
w.start()

## 组合多因子
lc = data.copy()
num_list = [42,44,36,15,4]
num_list = [i-1 for i in num_list]
m = []
alpha_m = lc[num_list[0]].copy()
for i in num_list[1:]:
    alpha_m['alpha'+str(i+1)] = lc[i]['alpha'+str(i+1)]

## 因子数据预处理
# 因子相关系数热力图
fig=plt.figure(figsize=(10,6))
relations= alpha_m.corr()
sns.heatmap(relations,annot=True,linewidths=0.05,linecolor='white',annot_kws={'size':8,'weight':'bold'})

# 中位数去极值
def extreme_process_MAD(sample):  # 输入的sample为时间截面的股票因子df数据
    factor_name = list(sample.columns)
    for name in factor_name:
        x = sample[name]
        median = x.median()
        MAD = abs(x - median).median()
        x[x>(median+3*1.4826*MAD)] = median+3*1.4826*MAD
        x[x<(median-3*1.4826*MAD)] = median-3*1.4826*MAD
        sample[name] = x
    return sample   
    
# 行业市值中性化
def data_scale_neutral(sample,date):
    stocks = list(sample.index)
    ind=w.wss(stocks, "industry_citic","unit=1;tradeDate="+date+";industryType=1", usedf=True)[1]
    Incap=w.wss(stocks, "val_lnmv","unit=1;tradeDate="+date+";industryType=1", usedf=True)[1]
    data_med = pd.get_dummies(ind,columns=['INDUSTRY_CITIC'])  # 生成0-1变量矩阵
    x = pd.concat([data_med,Incap],axis=1).dropna()
    X= np.array(x)
    sample = sample.loc[list(x.index)]
    factor_name = list(sample.columns)
    for name in factor_name:
        y = np.array(sample[name])
        if la.matrix_rank(X.T.dot(X)) == X.shape[1]:
            beta_ols = la.inv(X.T.dot(X)).dot(X.T).dot(y) # 最小二乘法计算拟合值
            residual = y - X.dot(beta_ols)  # 取残差为中性化后的因子值
        else:
            residual = y
            sample[name] = residual
    return sample   
    
# 标准化
def standardize(sample):
    factor_name = list(sample.columns)
    for name in factor_name:
        x = sample[name]
        sample[name] = (x - np.mean(x))/(np.std(x))
    return sample  
    
# 数据预处理
def data_process(sample,date):
    sample = extreme_process_MAD(sample)
    sample = data_scale_neutral(sample,date)
    sample = standardize(sample)
    return sample
   
# 回测筛选股票池
def get_stocks(trDate,A_stocks):
    status = w.wss(A_stocks, "trade_status,maxupordown,riskwarning,ipo_date", tradeDate=trDate, usedf=True)[1]
    date_least=w.tdaysoffset(-6,trDate,'Period=M').Data[0][0]   
    trade_codes=list(status[(status['TRADE_STATUS']=='交易')&(status['IPO_DATE']<=date_least)&(status['MAXUPORDOWN']==0)&(status['RISKWARNING']=='否')].index)    
    return trade_codes
      
# 最大化历史ICIR加权/历史IC均值加权
def IR_weight(bar_datetime_str,stocks,alpha_data):
    Period="W"
    begin_time = w.tdaysoffset(-12, bar_datetime_str, Period=Period, usedf=True).Data[0][0].strftime('%Y-%m-%d')
    time_list = w.tdays(begin_time,bar_datetime_str,Period=Period, usedf=True).Data[0]
    time_list = [time.strftime('%Y-%m-%d') for time in time_list]
    IC_s = []
    IC = []
    next_ret = w.wsd(stocks, "pct_chg", begin_time, bar_datetime_str, usedf=True, Period = Period)[1].fillna(value = 0).iloc[1:]
    for i in range(12):
        factor = alpha_data.loc[time_list[i]].reset_index().set_index(['codes']).drop(columns = ['date']).loc[stocks] # 转化为股票单层索引
        factor_name = list(factor.columns)
        ic_s = []
        for name in factor_name:
            try:
                ic_s.append(factor[name].corr(next_ret.iloc[i],method='spearman'))
            except:
                ic_s.append(factor[name].corr(next_ret.iloc[i-1],method='spearman'))
        IC_s.append(ic_s)
    IC_s = np.array(IC_s).T
    W = np.cov(IC_s)
    for i in range(len(IC_s)):
        IC.append(IC_s[i].mean())
    IC_IR = np.dot(W,np.array(IC))
    return IC_IR  # np.array(IC)
    
 # 最大化历史收益率加权
 def rate_weight(bar_datetime_str,stocks,alpha_data,num=10):
    Period="W"
    begin_time = w.tdaysoffset(-12, bar_datetime_str, Period=Period, usedf=True).Data[0][0].strftime('%Y-%m-%d')
    time_list = w.tdays(begin_time,bar_datetime_str,Period=Period, usedf=True).Data[0]
    time_list = [time.strftime('%Y-%m-%d') for time in time_list]
    IC_s = []
    IC = []
    next_ret = w.wsd(stocks, "pct_chg", begin_time, bar_datetime_str, usedf=True, Period = Period)[1].fillna(value = 0).iloc[1:]
    for i in range(12):
        factor = alpha_data.loc[time_list[i]].reset_index().set_index(['codes']).drop(columns = ['date']).loc[stocks]
        factor_name = list(factor.columns)
        ic_s = []
        for name in factor_name:
            g10 = list(factor.sort_values([name],ascending=False).iloc[:round(len(factor)/num)].index)
            try:
                ic_s = (next_ret.T.loc[g10]).T.iloc[i].mean()
            except:
                ic_s = (next_ret.T.loc[g10]).T.iloc[i-1].mean()
        IC_s.append(ic_s)
    IC_s = np.array(IC_s).T
    for i in range(len(IC_s)):
        IC.append(IC_s[i].mean())
    rate = np.array(IC)
    return rate
    
 # 因子打分
def factor_sum(sample,weight_list):
    factor_name = list(sample.columns)
    sample['alpha_sum'] = sample[factor_name[0]] * 0
    for i in range(len(factor_name)):
        sample['alpha_sum'] = sample['alpha_sum'] + sample[factor_name[i]] * weight_list[i]
    return sample
