"基于逻辑回归择时的指数增强策略"  

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
设置初始函数

聚宽的initialize()函数是策略全局函数，仅在回测最初运行一次，run_daily(func, time = 'every_bar')意思是将某个函数每天运行

注意有三个g.参数我放在了外面，因为聚宽的API手册说g.variable放在初始函数里自动成为全局变量，但是我发现只有g.security之类的特定变量才行，  
而自己定义的话是没法成为全局变量的，这种逻辑其实有点影响我的后续编写
```
g.t = 0
g.ts = 90
g.s = -1
def initialize(context):
    set_benchmark('000905.XSHG') #基准
    set_option('use_real_price', True) #用真实价格交易
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
                   type='stock') #交易费用
    run_daily(trade, time='every_bar') #每日运行trade函数
```
设置PEG因子：
PEG估值因子，公式：PEG=PE/（G*100）
分解：G为企业盈利增长率，PE=Price/EPS，得出PEG=Price/EPS*（G*100），因此PEG代表企业的股价与每股收益增长率比值
优势：在选用动态PE的基础上，PEG将兼具价值估值和成长估值的作用
*指数增强策略中用来更新股池的因子，一般选一个就够了，我比较喜欢融合PE和盈利成长的PEG*
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
接下来是
