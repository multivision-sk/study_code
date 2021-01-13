import cv2
import numpy as np


video_path = './video/bts_movie.mp4'
cap = cv2.VideoCapture(video_path)

output_size = (166,334) #width,height, 크기가 전체 화면을 넘어가는 경우 에러발생
fourcc = cv2.VideoWriter_fourcc('m','p','4','v') #코덱설정
out = cv2.VideoWriter('output.mp4' , fourcc, cap.get(cv2.CAP_PROP_FPS), output_size) #출력파일 이름 


if not cap.isOpened():
    exit()

tracker = cv2.TrackerCSRT_create() #CSRT object tracker


ret,img = cap.read()
cv2.namedWindow('select window')
cv2.imshow('select window', img) #첫번째 프레임에서

#setting ROI
rect = cv2.selectROI('select window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('select window',)

#initialize tracker
tracker.init(img,rect)


while True :
    ret,img = cap.read()
    if not ret :
        exit()

    success, box = tracker.update(img)

    left, top, w, h = [int(v) for v in box]

    center_x = left + w/2
    center_y =  top + h/2

    #저장할 영역
    result_top = int(center_y - output_size[1]/2)
    result_bottom = int(center_y + output_size[1]/2)
    result_left = int(center_x-output_size[0]/2)
    result_right = int(center_x+output_size[0]/2)

    result_img = img[result_top:result_bottom,result_left:result_right].copy()

    out.write(result_img)

    cv2.rectangle(img,pt1=(left,top), pt2=(left+w,top+h),color=(255,255,255),thickness=3)
    #ROI 그리기
    cv2.imshow('result img', result_img)
    cv2.waitKey(1)
    cv2.imshow('img',img)
    cv2.waitKey(1)



#ROI - region of interest 지정



