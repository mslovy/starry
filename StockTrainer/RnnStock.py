#coding=gbk
'''
Created on 2017��2��19��
@author: Lu.yipiao
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

#�������������������������������������������ݡ�������������������������������������������
df=pd.read_csv("000423_hist_data_new.csv")     #�����Ʊ����
data=np.array(df['high'])   #��ȡ��߼�����
data=data[::-1]      #��ת��ʹ���ݰ��������Ⱥ�˳������
#������ͼչʾdata
plt.figure()
plt.plot(data)
plt.show()
data_mean = np.mean(data)
data_std = np.std(data)
normalize_data=(data-data_mean)/data_std  #��׼��
normalize_data=normalize_data[:,np.newaxis]       #����ά��


#����ѵ����
#���ó���
time_step=20      #ʱ�䲽
rnn_unit=10       #hidden layer units
batch_size=60     #ÿһ����ѵ�����ٸ�����
input_size=1      #�����ά��
output_size=1     #�����ά��
lr=0.0001         #ѧϰ��
train_x,train_y=[],[]   #ѵ����
for i in range(len(normalize_data)-time_step-1):
    x=normalize_data[i:i+time_step]
    y=normalize_data[i+1:i+time_step+1]
    train_x.append(x.tolist())
    train_y.append(y.tolist()) 



#�������������������������������������������������������������������������������������
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #ÿ�������������tensor
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   #ÿ����tensor��Ӧ�ı�ǩ
#����㡢�����Ȩ�ء�ƫ��
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }



#�������������������������������������������������������������������������������������
def lstm(batch):      #��������������������Ŀ
    reuse_rnn = None
    #if batch == 1:
    #   reuse_rnn = True
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #��Ҫ��tensorת��2ά���м��㣬�����Ľ����Ϊ���ز������
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #��tensorת��3ά����Ϊlstm cell������
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit, reuse=reuse_rnn)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn�Ǽ�¼lstmÿ������ڵ�Ľ����final_states�����һ��cell�Ľ��
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #��Ϊ����������
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states



#������������������������������������ѵ��ģ�͡�����������������������������������
def train_lstm():
    global batch_size
    pred,_=lstm(batch_size)
    #��ʧ����
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #�ظ�ѵ��10000��
        step = 0
        for i in range(10000):
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                #ÿ10������һ�β���
            if step%100 == 0:
                print(i,step,loss_)
                saver.save(sess,os.path.join('stock_model', 'model.ckpt'))
            step+=1
        saver.save(sess, os.path.join('stock_model', 'model.ckpt'))
#train_lstm()

#��������������������������������Ԥ��ģ�͡���������������������������������������
def prediction():
    pred,_= lstm(1)      #Ԥ��ʱֻ����[1,time_step,input_size]�Ĳ�������
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #�����ָ�
        module_file = tf.train.latest_checkpoint('stock_model')
        saver.restore(sess, module_file)

        #ȡѵ�������һ��Ϊ����������shape=[1,time_step,input_size]
        predict_size = 5
        #prev_seq=[]
        #for i in range(predict_size):
        #prev_seq = train_x[-predict_size+i:-1]
        predict=[]
        #�õ�֮��100��Ԥ����
        #for i in range(predict_size):
        for i in range(predict_size):
            predict_one=sess.run(pred,feed_dict={X:[train_x[-predict_size+i]]})
            predict.append(predict_one[-1])
            #ÿ�εõ����һ��ʱ�䲽��Ԥ��������֮ǰ�����ݼ���һ���γ��µĲ�������
            #prev_seq=np.vstack((prev_seq[1:],next_seq[-1]))
        #������ͼ��ʾ���
        plt.figure()
        plt.plot(list(range(len(normalize_data))), normalize_data, color='b')
        plt.plot(list(range(len(normalize_data), len(normalize_data) + len(predict))), predict, color='r')
        predict_data = np.reshape(predict,(predict_size,1))
        print(predict_data * data_std + data_mean)
        plt.show()

prediction()