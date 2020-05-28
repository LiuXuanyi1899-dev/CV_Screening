"基于逻辑回归择时的机器学习策略"  
------------------------------
这是我在瑞银比赛的作品，当时手法生疏，策略尚不完善，但能一定程度上代表我的投资思路   
现在我又进行了一些优化，具体思路如下：   

通过logistic regression算法训练出决策边界线用来择时，与其他主流的机器学习策略不一样，我的思路是进行价格回归的捕捉（非价值的回归）  
具体操作如下：  
1.筛选股池：  
通过合成因子来追踪中证500和沪深300，找出综合得分前10的股票，半年更新一次
2.训练决策线：
首先获取股票的最近20个周线的最高价和最低价，其高低价格分别为两个特征值，用sigmod函数训练这两个特征值，将得到一条贯穿于股价波动周期的直线   
如果股价低于最近的回归值，可以判断股价超跌，因此买入，并一直持有直到盈利卖出，当然还会设置一些动态增减仓的函数

'''
注：此策略是在聚宽的内置编译器上创作的，没有搭建回测系统，主要原因是我不会安装zipline，非常麻烦，因此如果直接copy到软件上是没法运行的。同时一些代码没有优化（比如应该用向量化运算代替for循环等），导致运算时间较长，后续可以考虑优化一下
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
注意有几个g.参数我放在了外面，因为聚宽的API手册说g.variable放在初始函数里自动成为全局变量，但后来发现只有g.security之类的特定变量才行，  
自己定义的话是没法成为全局变量的，这种逻辑有点影响我的后续编写
```
g.t1 = 0
g.t2 = 0
g.ts = 180
g.s = -1
def initialize(context):
    set_benchmark('000905.XSHG') #基准
    set_option('use_real_price', True) #用真实价格交易
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock') #交易费用
    run_daily(trade, time='every_bar') #每日运行trade函数
```

第二步：设置均值计算函数：
这个函数用来计算合成因子中的四个均值，即沪深300和中证500所有成分股的总市值、PEG、PE、PB均值
```
def avg_calc():
    stk1 = get_index_stocks('000300.XSHG')
    stk2 = get_index_stocks('000905.XSHG')
    allstocks = stk1 + stk2 #设置的股票池
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
Point = (S[0] / avg_cap * w1 + S[1] / S[2] / avg_peg * w2 + S[3] / avg_pb * w3 + S[4] / avg_pcf * w4)/50   
S是函数的变量，在引用该计算函数时会使用一个dataframe，其包含所有中证500和沪深300的成分股，以及其总市值、PEG、PE、PB四个因子    
将四个因子与行业均值的比值乘上一定权重，再加总得到该股票的总分，分数越小越好，并筛选出综合得分前50的股票  
```
def point_calc(S):
    avg_cap, avg_peg, avg_pb, avg_pcf = avg_calc()
    w1 = 1
    w2 = 1
    w3 = 1
    w4 = 1
    return (S[0] / avg_cap * w1 + S[1] / S[2] / avg_peg * w2 + S[3] / avg_pb * w3 + S[4] / avg_pcf * w4) / 50
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
        df = df.sort_index(by='point').head(10) #以分数切片，获得前10个
        code_weight = dict(df['code':'point']) #打包成字典
        return code_weight  
        g.t = 1
    else:
        g.t += 1

```


5.设置逻辑回归函数:    

选逻辑回归的原因一是简单，二是可以很好地防止过拟合，算量也比较小，不会出现跑完聚宽的免费时间也回测不完的情况  
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

        # 高低价矩阵
        # 插入标签 0对应最低价，1对应最高价
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

        # 逻辑回归的X矩阵需要前插全是数字1的一列，我也是后来才知道可以用ones直接生成
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
                # W_update
                H = 1 / (1 + np.exp(-xmat * W)) #用sigmod来训练
                dw = xmat.T * (H - ymat)
                W -= alpha * dw
            return W  

        W = w_calc(0.001, 8000)  #在这里可以调节逻辑回归的参数，数据量少的话迭代一万次够了，后边都是边际递减，没什么效果
        w0 = W[0, 0]
        w1 = W[1, 0]
        w2 = W[2, 0]
        plotx1 = np.arange(0, 30, 0.01)
        plotx2 = -w0 / w2 - w1 / w2 * plotx1[-1]  #只要最近一天的那个值
        logistic_value_list.append(plotx2)
    return logistic_value_list #返回一个list   
``` 

接下来就到了最关键的一步，也是最让人头疼的一步，设置交易函数
为了能使用一个trade函数就完成任务，需要把90天运行一次的if函数放在里面，但是后来又发现聚宽的run_day()函数并不继承前一天运行时计算出来的变量，  
为了不让逻辑回归每天都跑一次（一来计算量太大，二来违背了策略的思想），只好把它放到第一函数里边，
```
def trade(context):
    code_weight = check_list()  从alpha_to_list() #获得购买列表
    logistic_value_list = logistic_regression()   #获得筛选的10支股票的逻辑回归值

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
完工，看一下回测结果   

![pic]https://github.com/LiuXuanyi1899-dev/CV_Screening/blob/master/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20200529001831.png
