import tensorflow as tf
import cv2
import numpy as np
import detect_face as df

tf.reset_default_graph()

labels = np.load('labels.npz')
name = labels['name']

sess = tf.Session()
saver = tf.train.import_meta_graph('./model/train_model.meta')
saver.restore(sess,tf.train.latest_checkpoint('./model/'))

graph = tf.get_default_graph()

x_sample = graph.get_tensor_by_name('x_sample:0')
y_label = graph.get_tensor_by_name('y_label:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
y_new = graph.get_tensor_by_name('y:0')

m_y = np.float32(np.zeros((1,10)))

cap = cv2.VideoCapture(0)
res, img = cap.read()
if res == True:
    cv2.imshow('Capture', img)
    cv2.waitKey(10000)
    cv2.imwrite('./m_lgh.jpg', img)
    img_gray = df.detection(img)
    cv2.imshow('img_gray222', img_gray)
    k = cv2.waitKey(10000)
    img_gray = np.float32(img_gray).reshape((1,32*32))
    img_gray = img_gray/255
    pre = sess.run(y_new, feed_dict={x_sample:img_gray, y_label:m_y, keep_prob:1.0})
    max_index = np.argmax(pre)
    m_name = name[max_index]
    
cv2.destroyAllWindows()  #删除建立的全部窗口
cap.release()            #关闭调用的摄像头
print("name:", m_name)















