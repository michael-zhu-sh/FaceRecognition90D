"""

【人脸识别90天】第1课：从摄像头视频流中捕获人脸图像，并保存彩色PNG文件。
Author: michael zhu
学号：94606
QQ:1265390626
email:michael.ai@foxmail.com

"""

from imutils.video import VideoStream
import argparse
import imutils
import time
import os
import cv2

#接收并分析命令行参数。
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--face', required=True,  help='file to load face cascade model.')    #检测人脸的模型文件。
ap.add_argument('-e', '--eye',  required=True,  help='file to load eye cascade model.')     #检测眼睛的模型文件。
ap.add_argument('-o', '--output',  required=True, help='directory to write face colorful png files.')   #保存人脸PNG文件的目录。
args = vars(ap.parse_args())
# 根据命令行参数指定的模型文件初始化haar级联分类器。
face_detector= cv2.CascadeClassifier(args['face'])
eye_detector = cv2.CascadeClassifier(args['eye'])

# 初始化摄像头和视频流捕获。
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2)   #延时2秒，等待摄像头硬件启动。
totalSaved=0    #已经保存的人脸图像数量。
while True:
    frame = vs.read()    #从视频流中捕获1帧图像。
    print('type of frame is:',type(frame))
    frame = imutils.resize(frame, width=512)    #缩小图像尺寸，便于加速人脸检测。
    colorImg = frame.copy() #原始彩色图像。
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); #彩色图像转化为灰度图像。
    #在灰度图像中检测人脸（可能有多个）。
    print('type of gray is:',type(gray))
    roi = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    print('type of roi is:',type(roi))
    #确定帧图像中是否为单人。
    if len(roi) == 0:
        print('没有在摄像头照射的区域范围内捕捉到人脸')
    elif len(roi)==1:
        print('成功捕捉到一张人脸，坐标[x=',roi[0][0],',y=',roi[0][1],']')
        x = roi[0][0]
        y = roi[0][1]
        w = roi[0][2]
        h = roi[0][3]
        #在检测出的单一人脸周围画红方框。
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        #在人脸范围中检测眼睛，并画出绿色方框。
        gray_roi = gray[y:y+h, x:x+w]
        eyes = eye_detector.detectMultiScale(gray_roi)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame, (x+ex,y+ey), (x+ex+ew,y+ey+eh), (0,255,0), 2)
    else:
        print('请确保摄像可以捕捉的区域内最多一人的情况下，按任意键继续执行人脸数据库采集操作\n')

    #显示彩色帧图像。
    cv2.imshow('VideoFrame', frame)
    key = cv2.waitKey(30) & 0xFF    #保持30FPS的采样速率。
    if key==ord('q'):   #按下q键退出本程序。
        break
    elif key==ord('s') and len(roi)==1: #按下s键保存彩色人脸图像。
        filename = os.path.sep.join([args["output"], "{}.png".format(str(totalSaved).zfill(5))])
        cv2.imwrite(filename, colorImg[y:y+h,x:x+w])    #用Numpy数组对图像像素进行访问时，应该先写图像高度所对应的坐标，再写图像宽度对应的坐标。
        totalSaved += 1
cv2.destroyAllWindows()
vs.stop()
    