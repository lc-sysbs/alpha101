from WindPy import *
from datetime import *
from WindAlgo import *
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import numpy.linalg as la
w.start()

alpha_data = alpha_m
    
def initialize(context):            
    context.capital = 10000000        # 回测的初始资金
    context.securities = w.wset("sectorconstituent", "date=20140101;windcode=000906.SH").Data[1]
    context.start_date = "20140104"  # 回测开始时间
    context.end_date = "20190530"    # 回测结束时间
    context.commission = 0.0003      # 手续费
    context.alpha_data = alpha_data 
    context.benchmark = '000906.SH'  # 设置回测基准

def handle_data(bar_datetime, context, bar_data):
    pass
    
def my_schedule1(bar_datetime, context, bar_data): 
    bar_datetime_str = bar_datetime.strftime('%Y-%m-%d')
    stock800 = w.wset("sectorconstituent", "date="+bar_datetime_str+";windcode=000906.SH").Data[1]
    stocks = get_stocks(bar_datetime_str,stock800)  # 获取筛选后的股票池
    ## 提取因子数据法
    data = context.alpha_data.loc[bar_datetime_str].reset_index().set_index(['codes']).drop(columns = ['date']) #提取因子数据并调整结构
    stock_v = list(set(stocks) & set(data.index))
    data = data.loc[stock_v].dropna(axis=0)  # 筛选出有效股票池对应数据
    profit = w.wss(stock_v, "fa_oigr_ttm,pe_ttm", tradeDate=bar_datetime_str, usedf=True)[1] # 加入财务因子
    profit.index = data.index
    data['1/PEG'] = profit['FA_OIGR_TTM']/profit['PE_TTM']
    data = data_process(data,bar_datetime_str)  # 数据预处理
    weight = []  # 因子加权权重
    data = factor_sum(data,weight) # 因子打分
    ## 回测中计算法
#     start_time = w.tdaysoffset(-5, bar_datetime_str,usedf=True).Data[0][0].strftime('%Y-%m-%d')
#     close = w.wsd(stocks,'close',start_time,bar_datetime_str,usedf = True)[1].reset_index().drop(columns = ['index'])
#     vwap = w.wsd(stocks,'vwap',start_time,bar_datetime_str,usedf = True)[1].reset_index().drop(columns = ['index'])
#     alpha = {'alpha':alpha42(vwap,close).iloc[-1]}
#     data = pd.DataFrame(alpha,index=stocks)
    
    data = data.sort_values([data.columns.values[-1]],ascending=False) # 按打分大小排序
    code_list = list(data[:round(len(stocks)/5)].index)  # 选出top组股票
    wa.change_securities(code_list) # 改变证券池 
    context.securities = code_list    
    list_sell = list(wa.query_position().get_field('code')) # 获取当前仓位股票池
    for code in list_sell:
        if code not in code_list:
            volumn = wa.query_position()[code]['volume'] # 找到每个股票的持仓量 
            res = wa.order(code,volumn,'sell',price='close', volume_check=False) 

def my_schedule2(bar_datetime, context,bar_data):
    buy_code_list=list(set(context.securities)-(set(context.securities)-set(list(bar_data.get_field('code')))))
    list_now = list(wa.query_position().get_field('code')) # 获取当前仓位股票池
    for code in buy_code_list:
        if code not in list_now:
            res = wa.order_percent(code,1/len(buy_code_list),'buy',price='close', volume_check=False)  # 等权买入

wa = BackTest(init_func = initialize, handle_data_func=handle_data)  # 实例化回测对象
wa.schedule(my_schedule1, "w", 0)   # w表示在每周执行一次策略，0表示偏移，表示月初第一个交易日往后0天
wa.schedule(my_schedule2, "w", 0) 
res = wa.run(show_progress=True)    # 调用run()函数开始回测,show_progress可用于指定是否显示回测净值曲线图
nav_df = wa.summary('nav')       # 获取回测结果，回测周期内每一天的组合净值
