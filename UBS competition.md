"基于合成因子选股的机器学习策略"  
------------------------------
这是我曾在瑞银比赛中使用过的思路，希望后续有富余时间去加入向量正则化运算，缩减代码行数，优化效率   

策略思路：
通过财务因子选股，logistic regression训练决策边界线来择时，继而进行价格回归的捕捉（非价值回归）  
1.筛选股池：  
通过合成因子来追踪中证500和沪深300，找出综合得分前10的股票，半年更新一次    
2.训练决策线：   
首先获取股票的最近20个周线的最高价和最低价，其高低价格分别为两个特征值，用sigmod函数训练这两个特征值，将得到一条贯穿于股价波动周期的直线   
如果股价低于最近的回归值，可以判断股价超跌，因此买入，并一直持有直到盈利卖出，当然还会设置一些动态增减仓的函数     

'''    
注：此策略在聚宽内置编译器上创作，没有搭建回测系统（我不会安装zipline，总是出现各种各样的问题）
'''

策略代码如下：
_____________________________________________________________________________________________________________________________________
导入需要的库，数据接口用的聚宽，即jqdatasdk（需要在auth处输入自己的账号密码，我就不写出来了）

```python
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from jqdatasdk import *
auth('账户','密码')
```

第一步：设置初始函数  
聚宽的initialize()是全局函数，仅在回测最初运行一次，run_daily(func, time = 'every_bar')意思是本策略的func每天运行一次  
有几个g.代表global，也就是策略的全局变量
```

def initialize(context):
    set_benchmark('000905.XSHG') #基准
    set_option('use_real_price', True) #用真实价格交易
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock') #交易费用
    run_daily(trade, time='every_bar') #每日运行trade函数
    g.days = 0
    g.refresh_rate = 180
    g.s = -1  #g.为全局变量
```

第二步：设置均值计算函数：
这个函数用来计算合成因子中的四个均值，即沪深300和中证500所有成分股的总市值、PEG、PE、PB均值
```
def avg_calc():
    stk1 = get_index_stocks('000300.XSHG') #沪深300
    stk2 = get_index_stocks('000905.XSHG') #中证500
    allstocks = stk1 + stk2
    # 获取股票池所有需要的数据
    df_all = get_fundamentals(query(
        valuation.code,
        valuation.market_cap,
        valuation.pe_ratio,
        indicator.inc_net_profit_year_on_year,
        valuation.pb_ratio,
        valuation.pcf_ratio,
    ).filter(
        valuation.code.in_(allstocks),
        valuation.code > 0,
        valuation.market_cap > 0,
        valuation.pe_ratio > 0,
        indicator.inc_net_profit_year_on_year > 0,
        valuation.pb_ratio > 0,
        valuation.pcf_ratio > 0,
    )).dropna()
    # 算出均值
    avg_cap = df_all['market_cap'].mean()
    avg_peg = df_all['pe_ratio'].mean() / df_all['inc_net_profit_year_on_year'].mean()
    avg_pb = df_all['pb_ratio'].mean()
    avg_pcf = df_all['pcf_ratio'].mean()
    # print(
    #     avg_cap,
    #     avg_peg,
    #     avg_pb,
    #     avg_pcf,
    # )
    #返回四个值
    return avg_cap, avg_peg, avg_pb, avg_pcf

```

3.打分函数：
自变量S在后面的交易函数中会调用，我设计的函数为：   
Point = S[0] / avg_cap * w1 + S[1] / S[2] / avg_peg * w2 + S[3] / avg_pb * w3 + S[4] / avg_pcf * w4  
S是函数的变量，在引用该计算函数时会使用一个dataframe，其包含所有中证500和沪深300的成分股，以及其总市值、PEG、PE、PB四个因子    
无数研究表明换手率、市值、PE在A股历史中的表现极佳，也是barra多因子框架的重要成分（换手率是动量策略中的因子，所以暂不考虑）   
将四个因子与行业均值的比值乘上一定权重，再加总得到该股票的总分，分数越小越好，并筛选出综合得分前50的股票  
```
def point_calc(S):
    avg_cap, avg_peg, avg_pb, avg_pcf = avg_calc()
    w1 = 1
    w2 = 1
    w3 = 1
    w4 = 1 #权重默认设置为1
    return S[0] / avg_cap * w1 + S[1] / S[2] / avg_peg * w2 + S[3] / avg_pb * w3 + S[4] / avg_pcf * w4
    # 此公式即为股票的分数
```
4.获得我们想要的股票：   
将所有股票代码和其对应分数保存在一个字典里  
```
def check_list(context):
    if g.t % g.ts == 0: #每180天运行一次
    
        stk1 = get_index_stocks('000300.XSHG')
        stk2 = get_index_stocks('000905.XSHG')
        allstocks = stk1 + stk2

        # 股票池
        df = get_fundamentals(query(
            valuation.code,
            valuation.market_cap,
            valuation.pe_ratio,
            indicator.inc_net_profit_year_on_year,
            valuation.pb_ratio,
            valuation.pcf_ratio,
        ).filter(
            valuation.code.in_(allstocks),
            valuation.code > 0,
            valuation.market_cap > 0,
            valuation.pe_ratio > 0,
            indicator.inc_net_profit_year_on_year > 0,
            valuation.pb_ratio > 0,
            valuation.pcf_ratio > 0,
        )).dropna()

        df['point'] = df[['market_cap', 'pe_ratio', 'inc_net_profit_year_on_year', 'pb_ratio', 'pcf_ratio']] \
            .T.apply(point_calc)  #调用上面那个打分函数
        df = df.sort_index(by='point').head(10) 取排名前10的股票
        stocks = list(df['code'])
        points = list(round(df['point'],3))
        code_weight = dict(zip(stocks, points)) 打包为字典，包含code和points两个key
        return code_weight  
        g.t = 1
    else:
        g.t += 1 #时间变量

```


5.设置逻辑回归函数:    

选逻辑回归的优势是欠拟合，算量小（相较于神经网络）  
```
def logistic_regression():
    logistic_value_list = []
    code_weight = check_list()
    for i in code_weight:
        df = get_bars(i, 20, unit='1w', fields=['high', 'low'], include_now=False, end_dt=None, fq_ref_date=None,
                      df=True)
        dflist = df.values.tolist()
        dfmat = np.mat(dflist)
        high1 = dfmat[:, 0]
        low1 = dfmat[:, 1]
        data0 = np.vstack((high1, low1))

        # 含高低价的矩阵
        # 插入标签：0对应最高价，1对应最低价
        def process_panel():
            i = 0
            k = [[0]]
            z = [[1]]
            panelh = [[0]]
            panell = [[1]]
            number = len(high1) - 1
            while i < number:
                i += 1
                panelh = np.vstack((panelh, k))
                panell = np.vstack((panell, z))
            return panelh, panell

        panelh, panell = process_panel()
        interpanel = np.append(panelh, panell, axis=0)

        # 插入时间序列，即一个数列n
        def process_time():
            t = 1
            i = 0
            timeline = [[1]]
            number = len(high1) - 1
            while i < number:
                i += 1
                t += 1
                tmat = [[t]]
                timeline = np.append(timeline, tmat, axis=0)
            return timeline

        timeline = process_time()
        timelinex = np.vstack((timeline, timeline))

        # X矩阵需要前插全是数字1的一列，我也是后来才知道可以用ones直接生成
        def count_1():
            number = len(high1) * 2 - 1
            i = 0
            k = [[1]]
            init1 = [[1]]
            while i < number:
                i += 1
                init1 = np.vstack((init1, k))
            return init1 #得到一个长度与导入数据相同的1数字列

        m1 = count_1()

        xmat1 = np.append(m1, timelinex, axis=1)
        xmat = np.hstack((xmat1, data0))
        ymat = interpanel
        #整合为xmat和ymat

        # logistic regression :计算W的值
        def w_calc(alpha=0.001, maxiter=1000): #alpha是学习率，maxiter为最大迭代次数
            W = np.mat(np.random.randn(3, 1)) #初始化3个W
            w_save = []
            for i in range(maxiter):
                # 选择sigmod来训练W值
                H = 1 / (1 + np.exp(-xmat * W))
                dw = xmat.T * (H - ymat)
                W -= alpha * dw
            return W  三行一列的矩阵

        W = w_calc(0.001, 10000)  #输入算法参数，学习率0.001，迭代一万次就差不多了
        w0 = W[0, 0]
        w1 = W[1, 0]
        w2 = W[2, 0]
        plotx = -w0 / w2 - w1 / w2 * 20  #只要最近一天的那个值
        logistic_value_list.append(plotx)
    return logistic_value_list #返回一个list   
``` 

由于run_daily()函数并不继承前一天运行时计算出来的变量，为了能使用一个trade函数就完成任务，需要把180天运行一次的if语句放在单独设置的更新函数
```
def update_func() :
    logistic_value_list = []
    code_weight = {}
    if g.days % g.refresh_rate == 0: #每180天运行一次
        logistic_value_list, code_weight = logistic_regression()
        g.days = 1
        return logistic_value_list, code_weight
    else :
        g.days += 1
```


最后一步，设置交易函数     
```
def trade(context):
    code_weight, logistic_value_list= update_func()  #获得购买列表、回归值

    for i code_weight.keys() :  #遍历我们的list
        totalcash = context.portfolio.starting_cash  #账户总资金
        avgcost = context.portfolio.positions[i].avg_cost  第i个股票的历史成本，没有的话应该是None值
        flag = 1  #一个开关，防止后面重复下单
        close_data = get_bars(i, 2, '1d', ['close'])  #获得'close'即收盘价
        current_price = close_data['close'][-1]  #获得i股票的最近收盘价，这两句不能合并，匪夷所思
        returning = current_price / avgcost - 1  #当前这支股票的收益率
        g.s += 1 
        v_dynamic = ((logistic_value_list[g.s])-current_price)/current_price  #这个是现价与回归值的偏移百分比
        position = code_weight[stock] * context.portfolio.total_value
        
        # 建仓
        if stock not in context.portfolio.positions:
            log.info('buy ', stock)
            order_target_value(stock, position)
            flag = 0
        if current_price < logistic_value_list[g.s]:
            order_value(i, v_dynamic) #动态控制仓位
        elif returning > 0.2 and flag == 0 : #止盈
            order_target_value(i, 0)
            flag = 1

 
    #检查180天更新的股池，有出局的股票就可以清掉
    for stock in context.portfolio.positions.keys():
        if stock not in code_weight:
            log.info('sell out ', stock)
            order_target(stock, 0)
    g.s = -1
```    
完工！回测结果如下：   

![pic](https://github.com/LiuXuanyi1899-dev/CV_Screening/blob/master/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20200529001831.png)
