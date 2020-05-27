"基于逻辑回归择时的指数增强策略"  
------------------------------
我在参加了瑞银的比赛后，开始学习量化策略编写（有python基础），此项目虽然尚不完善（非掉包，完全自己编写），但能一定程度上代表我的投资思路

注：此策略是在聚宽的内置编译器上创作的，没有搭建回测系统，主要原因是我不会安装zipline，非常麻烦，因此如果直接copy到软件上是没法运行的。同时一些代码没有优化（比如应该用向量运算代替for循环等），导致运算时间较长，后续可以考虑优化一下

策略代码如下：
_____________________________________________________________________________________________________________________________________
导入需要的库，数据接口我用的聚宽，即jqdatasdk（需要在auth处输入自己的账号密码）
这里要注意，jqdatasdk在聚宽的编译器上没法运行，只能用jqdata库，而且不需要激活账号权限，所以在聚宽编写的话直接import jqdata就行了
```python
import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from jqdatasdk import *
auth('账户','密码')
```

第一步：设置初始函数  
聚宽的initialize()是全局函数，仅在回测最初运行一次，run_daily(func, time = 'every_bar')意思是本策略的func每天运行一次  
注意有几个g.参数我放在了外面，因为聚宽的API手册说g.variable放在初始函数里自动成为全局变量，但后来发现只有g.security之类的特定变量才行  
而自己定义的话是没法成为全局变量的，这个逻辑其实有点影响我的后续编写
```
g.t1 = 0
g.t2 = 0
g.ts = 90
g.s = -1
def initialize(context):
    set_benchmark('000905.XSHG') #基准
    set_option('use_real_price', True) #用真实价格交易
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock') #交易费用
    run_daily(trade, time='every_bar') #每日运行trade函数
```

第二步：设置PEG因子：
PEG估值因子，公式：PEG=PE/（G*100）
分解：G为企业盈利增长率，PE=Price/EPS，得出PEG=Price/EPS*（G*100），因此PEG代表企业的股价与每股收益增长率比值  
优势：在选用动态PE的基础上，PEG将兼具价值估值和成长估值的作用  
指数增强策略中用来更新股池的因子，一般选一个就够了，我比较喜欢融合PE和盈利成长的PEG
```
def Alpha_PEG():
    # query股池里所有的股票
    stock_list = get_index_stocks('000300.XSHG')  # 选择沪深300为初始指数
    q_PE_G = query(valuation.code, valuation.pe_ratio, indicator.inc_net_profit_year_on_year).filter(valuation.code.in_(stock_list))
    # 返回一个dataframe：code、PE、G；默认的date位置参数 = context.current_dt，即前一天

    # 通过get_fundamentals函数获得上述q_PE_G的dataframe数据，保存为de_PE_G,其内容为code、pe、G的三列df
    df_PE_G = get_fundamentals(q_PE_G)
    # 去掉PE、G为负的股票
    df_Growth_PE_G = df_PE_G[(df_PE_G.pe_ratio >0)&(df_PE_G.inc_net_profit_year_on_year >0)]

    df_Growth_PE_G.dropna()    # 清洗空缺值
    Series_PE = df_Growth_PE_G.loc[:,'pe_ratio']    # 用Series存放PE
    Series_G = df_Growth_PE_G.loc[:,'inc_net_profit_year_on_year']    # Series存放G
    Series_PEG = Series_PE/Series_G    # 计算PEG
    Series_PEG.index = df_Growth_PE_G.iloc[:,0]
    df_PEG = pd.DataFrame(Series_PEG)
    # df_PEG.sort_values(by =  axis= 0, ascending= True)
    return df_PEG
    # 返回df_PEG这一dataframe
```

接下来将上个函数，即含有PEG的dataframe进行一些处理，并输出一个list：
```
def alpha_to_list():
    if g.t % g.ts == 0:
        num_stocks = 10  #每g.ts天，也就是90更新一次buy_list
        buy_list = []    # 得到一个dataframe：index为股票代码，data为相应的PEG值
        # 将满足alpha1的股票存入buy_list，run_monthly
        df_PEG = alpha_PEG()  # 将股票按PEG升序排列，返回daraframe
        try:
            df_sort_PEG = df_PEG.sort(columns=[0], ascending=[1])
        except AttributeError:
            df_sort_PEG = df_PEG.sort_values(by=[0], ascending=[1])
        for i in range(num_stocks):
            buy_list.append(df_sort_PEG.index[i])
        return buy_list
        g.t = 1
    else:
        g.t += 1
        #现在我们有了一list的股票代码
```

第三步：设置逻辑回归函数  
选用逻辑回归的原因一是简单，二是可以很好地防止过拟合，一根简简单单的直线很适合择时  
```
def logistic_regression():
    if g.t2 % g.ts == 0:
    
        logistic_value_list = []
        buy_list = alpha_to_list()
        for i in buy_list: #这里用了for loop，要将
            df = get_bars(i, 20, unit='1w', fields=['high', 'low'], include_now=False, end_dt=None, fq_ref_date=None,
                          df=True)
            dflist = df.values.tolist()
            dfmat = np.mat(dflist)
            high1 = dfmat[:, 0]
            low1 = dfmat[:, 1]
            data0 = np.vstack((high1, low1))

            # 高低价矩阵
            # 插入标签
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

            # 插入时间序列
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

            # 逻辑回归的X矩阵需要前插全是数字1的一列
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
            plotx2 = -w0 / w2 - w1 / w2 * plotx1[-1]  #只算最近一天的那个值
            logistic_value_list.append(plotx2)
        return logistic_value_list #返回一个list
        g.t2 = 1
    else :
        g.t2 += 1        
``` 

接下来就到了最关键的一步，也是最让人头疼的一步，设置交易函数
为了能使用一个trade函数就完成任务，需要把90天运行一次的if函数放在里面，但是后来又发现聚宽的run_day()函数并不继承前一天运行时计算出来的变量，  
为了不让逻辑回归每天都跑一次（一来计算量太大，二来违背了策略的思想），只好把它放到第一个alpha_to_list()函数里边，
```
def trade(context):
    buy_list = alpha_to_list()  从alpha_to_list() #获得购买列表
    logistic_value_list = logistic_regression()   #获得筛选的10支股票的逻辑回归值

    for i in buy_list :  #遍历我们的buy_list
        totalcash = context.portfolio.starting_cash  #账户总资金
        avgcost = context.portfolio.positions[i].avg_cost  第i个股票的历史成本，没有的话应该是None
        flag = 1  #一个开关，防止后面重复下单
        close_data = get_bars(i, 2, '1d', ['close'])  #获得'close'即收盘价
        current_price = close_data['close'][-1]  #获得i股票的最近收盘价，这两句不能合并，匪夷所思
        returning = current_price / avgcost - 1  #当前这支股票的收益率
        g.s += 1 
        v_dynamic = ((logistic_value_list[g.s])-current_price)/current_price  #这个是现价与回归值的偏移百分比
        
        if current_price < logistic_value_list[g.s] and context.portfolio.positions[i].avg_cost == 0: 
            order_value(v_dynamic, totalcash/15)           #如果该股票现价低于回归值没有仓位，建仓1/15资金
        elif current_price < logistic_value_list[g.s]:     #建仓后股价波动会更新v_dynamic，此时调整仓位
            holding_rate = context.portfolio.positions_value[i]*10/totalcash
            order_value(i, (v_dynamic - holding_rate)*totalcash/10)
        else returning > 0.2 and flag == 0 : #赚了20%就止盈，但是跌不止损
            order_target_value(i, 0)
            
    for stock in context.portfolio.positions : #检查90天更新的股池，有出局的股票就可以清掉
        if stock not in buy_list:
            order_target_value(stock, 0)
            
    g.s = -1
```    
完工，看一下回测结果  

