## 分层测试
def return_fenxi(df,d,num,alpha = 'alpha'):
    df.fillna(value = 0,inplace = True)
    date_ = d
    return_s = []
    for i in range(len(date_)):
        stock_v = stock_valid_df.loc[d[i]].dropna().values
        stock_v = list(set(stock_v) & set(df.loc[d[i]].index))
        dff = df.loc[d[i]].loc[stock_v]
        x = dff[alpha]
        if x.sum() != 0:  # 筛选掉因子值异常期
            df_i = dff.sort_values(alpha)
            return_list = []
            for j in range(num):
                n1 = round(len(df_i)*j/num)
                n2 = round(len(df_i)*(j+1)/num)
                df_j = df_i.iloc[n1:n2]
                return_j = df_j['NEXT_RET'].mean()+1
                return_list.append(return_j)
            return_s.append(return_list)
    x = np.array(return_s).T  # 二维列表转化为二维数组转置
    return_s = [list(i) for i in x]
    for i in range(num):
        x = return_s[i]
        for j in range(1,len(x)):
            x[j] = x[j]*x[j-1]
        return_s[i] = x
    culmu = [re[-1] for re in return_s]
    return culmu,return_s

def fencengceshi(data,next_ret,num):
    data_re = []
    data_re1 = []
    for i in range(len(data)):
        lc = data[i]
        #lc.fillna(value = 0,inplace = True)
        d = list(next_ret.index.levels[0])
        lc = data[i].loc[d]
        lc['NEXT_RET'] = next_ret.NEXT_RET
        list_re1,list_re = return_fenxi(lc,d,num,alpha = data[i].columns[0])
        data_re.append(list_re)
        data_re1.append(list_re1)
        print(data_re1[i]) # 各因子最终各组累计收益
    return data_re

df = fencengceshi(data,next_ret,num=20) # 各因子分组累计收益序列
dff = fencengceshi(data,next_ret,num=1) # 计算等权基准线

## 各组收益时序图,top/middle/bottom
n = 1
x = df[n-1]
y1 = x[0]
y2 = x[num/2]
y3 = x[-1]
d = list(next_ret.index.levels[0])[-len(y1):]
bench = dff[1][0][-len(y1):]

plt.subplots(figsize=(15,5))  # 图的长宽设置
plt.plot(d,y1,label='Group1')
plt.plot(d,y2,label='Group2')
plt.plot(d,y3,label='Group3')
plt.plot(d,bench,label='ZZ800')
plt.legend()
plt.title('alpha'+str(n)+' 分层测试(月)')
plt.xlabel('回测区间')
plt.ylabel("净值")
