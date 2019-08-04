import cv2
import os

name = 'luoguanghui'
file_path = 'E:/faceImages/'
#若文件夹不存在，则创建文件夹
if not os.path.exists(file_path + name):
    os.makedirs(file_path+name)
cap = cv2.VideoCapture(0)
for i in range(600):
    f, frame = cap.read()  #此刻拍照
    cv2.imshow('Capture', frame)
    k = cv2.waitKey(1)
    #防止覆盖原有的照片
    if not os.path.exists(file_path + name + '/' + str(i) + '.jpg'):
        #将拍摄内容保存为jpg图片
        cv2.imwrite(file_path + name + '/' + str(i) + '.jpg', frame)
        print(i, '.jpg')

cv2.destroyAllWindows()  #删除建立的全部窗口
cap.release()            #关闭调用的摄像头