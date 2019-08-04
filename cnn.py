import tensorflow as tf
import numpy as np
from sklearn.preprocessing import OneHotEncoder

tf.reset_default_graph()
sess = tf.InteractiveSession()

#一、函数声明部分  
def weight_variable(shape):  
#正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0  
    initial = tf.truncated_normal(shape, stddev=0.1)  
    return tf.Variable(initial)  
def bias_variable(shape):  
#创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1  
    initial = tf.constant(0.1, shape=shape)  
    return tf.Variable(initial)  
def conv2d(x, w):    
#卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘  
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')    
def max_pool_2x2(x):    
#池化卷积结果（conv2d）池化层采用kernel大小为2*2，步数也为2，周围补0，取最大值。数据量缩小了4倍  
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 

data = np.load('data.npz')
labels = np.load('labels.npz')

train_data, test_data = data['train_data'], data['test_data']
train_labels, test_labels, name = labels['train_labels'], labels['test_labels'], labels['name']
train_data = np.float32(train_data)
test_data = np.float32(test_data)
train_labels = np.float32(train_labels).reshape((4800,1))
test_labels = np.float32(test_labels).reshape((1200,1))
#独热编码标签
ohe = OneHotEncoder()
ohe.fit(train_labels)
train_labels = np.float32(ohe.transform(train_labels).toarray())
ohe.fit(test_labels)
test_labels = np.float32(ohe.transform(test_labels).toarray())
#train_labels = np.float32(tf.one_hot(train_labels,depth=10,axis=1))
#test_labels = np.float32(tf.one_hot(test_labels,depth=10,axis=1))

x_sample = tf.placeholder(tf.float32, [None, 32*32], name = 'x_sample')
y_label = tf.placeholder(tf.float32, [None, 10], name = 'y_label')
x_image = tf.reshape(x_sample, [-1, 32, 32, 1])  #tf改成np

#第一层卷积、RELU、池化
w_conv1 = weight_variable([7, 7, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积、RELU、池化
w_conv2 = weight_variable([7, 7, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#第三层全连接
w_fc1 = weight_variable([8*8*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

#dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，个人觉得样本较少时很必要  
#使用占位符，由dropout自动确定scale，也可以自定义，比如0.5，根据tensorflow文档可知，程序中真实使用的值为1/0.5=2，也就是某些输入乘以2，同时某些输入乘以0  
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')   
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  #对卷积结果执行dropout操作

#第四层输出
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
m_y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2, name = 'y')

#定义损失函数，并优化
m_cross_entropy = -tf.reduce_sum(y_label * tf.log(m_y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(m_cross_entropy)
#模型保存
saver = tf.train.Saver()

#训练
correct_prediction = tf.equal(tf.argmax(m_y_conv, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.global_variables_initializer().run()
for j in range(50):
    print(j,":")
    for i in range(96):
        batch0 = train_data[50*i:50*(i+1),:]
        batch1 = train_labels[50*i:50*(i+1)]
        if i%12 == 0:
            train_accuracy = accuracy.eval(feed_dict={x_sample: batch0, y_label: batch1, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
        train_step.run(feed_dict={x_sample: batch0, y_label: batch1, keep_prob: 0.5})

#测试
test_accuracy = []
batch0 = test_data
batch1 = test_labels
test_accuracy.append(accuracy.eval(feed_dict={x_sample: batch0, y_label: batch1, keep_prob: 1.0}))
print("test accuracy %g"% np.mean(np.array(test_accuracy)))

#保存模型
saver.save(sess, './model/train_model')







