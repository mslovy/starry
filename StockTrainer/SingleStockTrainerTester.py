import tensorflow as tf
import pandas as pd
import numpy

hist_data = pd.read_csv("000423_hist_data_new.csv")
basics_data = pd.read_csv("000423_basics.csv", encoding="gbk")
num_test = 10

hist_data = hist_data.drop('date',axis=1)
hist_data = hist_data.drop('price_change', axis=1)
hist_data = hist_data.drop('turnover', axis=1)

highest_price = max(hist_data['high'])
lowest_price = min(hist_data['low'])

maxvol = max(hist_data['volume'])
lowvol = min(hist_data['volume'])

hist_data['open'] = (2*hist_data['open'] - (lowest_price + highest_price))/(highest_price - lowest_price)
hist_data['high'] = (2*hist_data['high'] - (lowest_price + highest_price))/(highest_price - lowest_price)
hist_data['close'] = (2*hist_data['close'] - (lowest_price + highest_price))/(highest_price - lowest_price)
hist_data['low'] = (2*hist_data['low'] - (lowest_price + highest_price))/(highest_price - lowest_price)
hist_data['ma5'] = (2*hist_data['ma5'] - (lowest_price + highest_price))/(highest_price - lowest_price)
hist_data['ma10'] = (2*hist_data['ma10'] - (lowest_price + highest_price))/(highest_price - lowest_price)
hist_data['ma20'] = (2*hist_data['ma20'] - (lowest_price + highest_price))/(highest_price - lowest_price)
hist_data['volume'] = (2*hist_data['volume'] - (lowvol + maxvol))/(maxvol - lowvol)
hist_data['v_ma5'] = (2*hist_data['v_ma5'] - (lowvol + maxvol))/(maxvol - lowvol)
hist_data['v_ma10'] = (2*hist_data['v_ma10'] - (lowvol + maxvol))/(maxvol - lowvol)
hist_data['v_ma20'] = (2*hist_data['v_ma20'] - (lowvol + maxvol))/(maxvol - lowvol)
hist_data['p_change'] = hist_data['p_change']/10

#print("open: ", hist_data['open'])
#print("volume: ", hist_data['volume'])

p_data = hist_data['p_change']
hist_data['pr_change'] = p_data[1:]
dpr_change = hist_data['pr_change']
dpr_change[0:dpr_change.size-1] = dpr_change[1:]
dpr_change[dpr_change.size-1] = 0

training_set = hist_data.tail(hist_data.shape[0] - num_test)
testing_set = hist_data.head(num_test)

# 超参数设定
learning_rate = 0.00001
training_epochs = 10000
batch_size = 50
display_step = 1
examples_to_show = 10

# 神经网络参数设定
n_hidden_1 = 20000 # 1st layer num features
n_hidden_2 = 10000 # 2nd layer num features
n_hidden_3 = 5 # 3nd layer num features
n_input = 12
n_output = 1

# tf 计算图输入
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None])

weights = {
    'w1': tf.get_variable("W1", shape=[n_input, n_hidden_1],initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
    'w2': tf.get_variable("W2", shape=[n_hidden_1, n_hidden_2],initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
#    'w3': tf.get_variable("W3", shape=[n_hidden_2, n_hidden_3],initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
    'w4': tf.get_variable("W4", shape=[n_hidden_2, n_output],initializer=tf.contrib.layers.xavier_initializer(uniform=False)),
}
biases = {
    'w1': tf.get_variable("b1", shape=[n_hidden_1],initializer=tf.contrib.layers.xavier_initializer(uniform=True)),
    'w2': tf.get_variable("b2", shape=[n_hidden_2],initializer=tf.contrib.layers.xavier_initializer(uniform=True)),
#    'w3': tf.get_variable("b3", shape=[n_hidden_3],initializer=tf.contrib.layers.xavier_initializer(uniform=True)),
    'w4': tf.get_variable("b4", shape=[n_output],initializer=tf.contrib.layers.xavier_initializer(uniform=True)),
}


 # 隐层使用sigmoid激励函数 #1
layer_1 = tf.nn.tanh(tf.add(tf.matmul(X, weights['w1']),
                                biases['w1']))

# 隐层使用sigmoid激励函数 #2
layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['w2']),
                                biases['w2']))

#layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['w3']),
#                                biases['w3']))

layer_4 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['w4']),
                                biases['w4']))



# 搭建模型
encoder_op = layer_4

# 预测
y_pred = encoder_op
# Targets (Labels) are the input data.
y_true = Y

#y_diff_true = tf.multiply(tf.abs(y_pred - y_true), 1/y_true)

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_sum(tf.pow(y_pred - y_true, 2))
#cost = tf.reduce_sum(y_diff_true)
#cost = tf.reduce_mean(-tf.reduce_sum(y_true*tf.log(y_pred) + (1-y_true)*tf.log(1-y_pred), reduction_indices=1))
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
# Using InteractiveSession (more convenient while using Notebooks)
sess = tf.InteractiveSession()
sess.run(init)

w1 = sess.run(weights['w1'])
w2 = sess.run(weights['w2'])
#w3 = sess.run(weights['w3'])
w4 = sess.run(weights['w4'])
b1 = sess.run(biases['w1'])
b2 = sess.run(biases['w2'])
#b3 = sess.run(biases['w3'])
b4 = sess.run(biases['w4'])
#print("W1:", w1, "W2:",w2, "W3:",w3, "W4:",w4, "b1:",b1, "b2:",b2, "b3:",b3, "b4:",b4)
#print("W1:", w1, "W2:",w2, "W4:",w4, "b1:",b1, "b2:",b2, "b4:",b4)

predict = sess.run(y_pred, feed_dict={X: training_set.iloc[:,:-1], Y: training_set.iloc[:,-1]})

#l1 = sess.run(layer_1, feed_dict={X: training_set.iloc[:,:-1], Y: training_set.iloc[:,-1]})
#print("l1: ", l1)

#l2 = sess.run(layer_2, feed_dict={X: training_set.iloc[:,:-1], Y: training_set.iloc[:,-1]})
#print("l2: ", l2)

#l3 = sess.run(layer_3, feed_dict={X: training_set.iloc[:,:-1], Y: training_set.iloc[:,-1]})
#print("l3: ", l3)

#l4 = sess.run(layer_4, feed_dict={X: training_set.iloc[:,:-1], Y: training_set.iloc[:,-1]})
#print("l4: ", l4)

print(predict)

