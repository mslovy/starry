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

p_data = hist_data['p_change']
hist_data['pr_change'] = p_data[:-1]
dpr_change = hist_data['pr_change']
dpr_change[1:] = dpr_change[:-1]
dpr_change[0] = 0

training_set = hist_data.tail(hist_data.shape[0] - num_test)
testing_set = hist_data.head(num_test)

# 超参数设定
learning_rate = 0.0000001
training_epochs = 10000
batch_size = 50
display_step = 1
examples_to_show = 10

# 神经网络参数设定
n_hidden_1 = 50000 # 1st layer num features
n_hidden_2 = 5000 # 2nd layer num features
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

# 搭建encoder
def encoder(x):
    # 隐层使用sigmoid激励函数 #1
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(x, weights['w1']),
                                   biases['w1']))
    layer_1 = tf.nn.dropout(layer_1, keep_prob=0.5)
    # 隐层使用sigmoid激励函数 #2
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, weights['w2']),
                                   biases['w2']))
    layer_2 = tf.nn.dropout(layer_2, keep_prob=0.5)

#    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['w3']),
#                                   biases['w3']))
#    layer_3 = tf.nn.dropout(layer_3, keep_prob=0.5)

    layer_4 = tf.nn.tanh(tf.add(tf.matmul(layer_2, weights['w4']),
                                   biases['w4']))
    #layer_final = tf.abs(tf.multiply(tf.subtract(layer_4 - y),0.5))
    return layer_4

# 搭建模型
encoder_op = encoder(X)

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

total_batch = int((training_set.shape[0])/batch_size)

def shuffe_data(data):
    perm = numpy.arange(data.shape[0])
    numpy.random.shuffle(perm)
    return data.iloc[perm,:]

def next_batch(data, index):
    start = batch_size * index
    end = batch_size * (index+1)
    data_x = data.iloc[start:end,:-1]
    data_y = data['pr_change'].iloc[start:end]
    return data_x, data_y

def predict_samples():
    testX = training_set.iloc[:, :-1]
    testY = training_set['pr_change']
    predict_result = sess.run(
        y_pred, feed_dict={X: testX})
    predict_result = numpy.reshape(predict_result,(predict_result.size,))
    #print(predict_result)
    #print(predict_result)
    #print(testY)
    predict_diff = abs(predict_result - testY)
    #print(predict_diff)
    best_predict = predict_diff[predict_diff < 0.15]
    print("samples accuracy: %s", best_predict.size / predict_diff.size)

def predict():
    testX = testing_set.iloc[1:, :-1]
    #print(testX)
    testY = testing_set['pr_change'].iloc[1:]
    predict_result = sess.run(
        y_pred, feed_dict={X: testX})
    #myw1 = sess.run(weights['w1'])
    #print(myw1)
    predict_result = numpy.reshape(predict_result,(predict_result.size,))
    print(predict_result)
    #print(testY)
    predict_diff = abs(predict_result - testY)
    #print(predict_diff.values)
    best_predict = predict_diff[predict_diff < 0.15]
    #print(predict_result)
    #print(testY.values)
    print("accuracy: %s", best_predict.size / predict_diff.size)

# Training cycle
for epoch in range(training_epochs):
    # Loop over all batches
    shuffed_data = shuffe_data(hist_data)
    for i in range(total_batch):
        batch_xs, batch_ys = next_batch(shuffed_data, i)
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
    # Display logs per epoch step
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
              "cost=", "{:.9f}".format(c))

    # prediction
    if epoch % 10 == 0:
        predict_samples()
        predict()

print("Optimization Finished!")



# Applying encode and decode over test set
predict()

# Compare original images with their reconstructions
#f, a = plt.subplots(2, 10, figsize=(10, 2))
#for i in range(examples_to_show):
#    a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
#    a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
#f.show()
#plt.draw()
#plt.waitforbuttonpress()
