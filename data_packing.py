import numpy as np
import os
import cv2

faceImagesGray_path = 'E:/faceImageGray'  #人脸检测后的数据文件路径
file_name = os.listdir(faceImagesGray_path)
data = []
labels = []
name = []
i = 0
for file in file_name:
    print("step", i, ":", file)
    name.append(file)
    for faceImage in os.listdir(faceImagesGray_path + '/' + file):
        img = cv2.imread(faceImagesGray_path + '/' + file + '/' + faceImage, 0)
        img = np.array(img)  #转成numpy的array
        img_new = img.reshape(1, 32*32)  #转成1*1024一行的形状
        data.append(img_new)
        labels.append(i)
    i += 1
    
data = np.array(data).reshape(6000, 1024)  #转成6000*1024的数组
data = data/255
labels = np.array(labels)  #转成numpy的array

index_tr, index_te = [], []
tr_size, n, m = 0.8, 600, 32  #训练集占比，每一类的图片个数，图像的大小
#打乱索引下标
for i in range(10):
    index = [j+i*n for j in range(n)]
    np.random.shuffle(index)
    index_tr.append(index[:int(n*tr_size)])
    index_te.append(index[int(n*tr_size):])
index_tr = np.array(index_tr)
index_te = np.array(index_te)
index_tr = index_tr.reshape(4800, 1)
index_te = index_te.reshape(1200, 1)
data_tr = data[index_tr, :].reshape(4800, 1024)
data_te = data[index_te, :].reshape(1200, 1024)
labels_tr = labels[index_tr].reshape(4800,)
labels_te = labels[index_te].reshape(1200,)

index1, index2 = [j for j in range(4800)], [j for j in range(1200)]
np.random.shuffle(index1)
np.random.shuffle(index2)
train_data, train_labels = data_tr[index1], labels_tr[index1]
test_data, test_labels = data_te[index2], labels_te[index2]

np.savez('./data.npz', train_data = train_data, test_data = test_data)
np.savez('./labels.npz', train_labels = train_labels, test_labels = test_labels, name = name)

    