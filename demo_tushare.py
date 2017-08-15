import tushare as ts


import tushare as ts
#ts.set_token('758bd93ed8a32b88dc2f9fcbdd4ce47a78c1af389498f393d48ee46b0b8c4d89')

#t = ts.Market()
#df = st.MktEqud(tradeDate='20170622', field='ticker,secShortName,preClosePrice,openPrice,highestPrice,lowestPrice,closePrice,turnoverVol,turnoverRate')
#df['ticker'] = df['ticker'].map(lambda x: str(x).zfill(6))
#print(df)

stock_info=ts.get_stock_basics()
for i in stock_info.index:
  print(i)
print(stock_info.iloc[0,:])
print(stock_info.shape)

#df = ts.get_hist_data('000423')
#print(df.index[0].split("-")[1] > df.index[1].split("-")[1])
#df.to_csv('000423_hist_data_new.csv')

#df = ts.get_stock_basics()
#df.ix['000423'].to_csv('000423_basics.csv')


#df_small = df.head(10)
#print(df_small  )
#print(max(df_small['high']))

#df1 = ts.get_today_all('sh', date='2014-01-19')
#print(df1)

#eq = ts.Equity()
#df = eq.Equ(equTypeCD='A', listStatusCD='L', field='ticker,secShortName,totalShares,nonrestFloatShares')
#df['ticker'] = df['ticker'].map(lambda x: str(x).zfill(6))

#df = ts.get_h_data('000423', start='2014-07-14')
#print(df)

#df = ts.get_stock_basics()
#print(df.ix['000423'])

#df = ts.get_tick_data(code='000423', date='2017-07-12')
#print(df)
#print(df.get_value('000423','name'))