import cv2
cap = cv2.VideoCapture('http://10.231.34.6:8080/video')
print('connected:', cap.isOpened())
cap.release()