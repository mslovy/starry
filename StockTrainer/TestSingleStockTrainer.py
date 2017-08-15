import tensorflow as tf
import pandas as pd
import numpy
import numpy as np

def shuffe_data(data):
    perm = numpy.arange(data.shape[0])
    numpy.random.shuffle(perm)
    return data.iloc[perm,:]

#A = tf.random_normal([1000, 1000], stddev=1/1000)
# Initializing the variables
#init = tf.initialize_all_variables()

# Launch the graph
# Using InteractiveSession (more convenient while using Notebooks)
#sess = tf.InteractiveSession()
#sess.run(init)

#r = sess.run(A)
#print(r)

hist_data = pd.read_csv("000423_hist_data_new.csv")
basics_data = pd.read_csv("000423_basics.csv", encoding="gbk")
num_test = 2

hist_data = hist_data.head(10)
#print(hist_data[::-1])
#for i in range(1, hist_data.shape[1]):
#    mydata = hist_data.iloc[:,i]
#    print(mydata.shape)
#print(hist_data.iloc[:,0])
hist_data = hist_data.drop('date',axis=1)
hist_data = hist_data.drop('price_change', axis=1)
new_data = hist_data['p_change']
hist_data['pr_change'] = new_data[:-1]
dpr_change = hist_data['pr_change']
dpr_change[1:] = dpr_change[:-1]
dpr_change[0] = 0
#print(hist_data.iloc[:,-1])
#print(numpy.c_[hist_data, new_data])
#print(hist_data[hist_data['p_change']>0.5])

#print(shuffe_data(hist_data))

highest_price = max(hist_data['high'])
lowest_price = min(hist_data['low'])

maxvol = max(hist_data['volume'])
lowvol = min(hist_data['volume'])

hist_data['open'] = (hist_data['open'] - lowest_price)/(highest_price - lowest_price)
hist_data['high'] = (hist_data['high'] - lowest_price)/(highest_price - lowest_price)
hist_data['close'] = (hist_data['close'] - lowest_price)/(highest_price - lowest_price)
hist_data['low'] = (hist_data['low'] - lowest_price)/(highest_price - lowest_price)
hist_data['ma5'] = (hist_data['ma5'] - lowest_price)/(highest_price - lowest_price)
hist_data['ma10'] = (hist_data['ma10'] - lowest_price)/(highest_price - lowest_price)
hist_data['ma20'] = (hist_data['ma20'] - lowest_price)/(highest_price - lowest_price)
hist_data['volume'] = (hist_data['volume']-lowvol)/(maxvol - lowvol)
hist_data['v_ma5'] = (hist_data['v_ma5']-lowvol)/(maxvol - lowvol)
hist_data['v_ma10'] = (hist_data['v_ma10']-lowvol)/(maxvol - lowvol)
hist_data['v_ma20'] = (hist_data['v_ma20']-lowvol)/(maxvol - lowvol)

#print(hist_data)

training_set = hist_data.tail(hist_data.shape[0] - num_test)
testing_set = hist_data.head(num_test)

#print(testing_set)


