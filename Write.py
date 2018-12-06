#!/usr/bin/env python3
#make sure opencv-python is installed locally
import numpy as np
import cv2
import argparse
from collections import deque
import random


cap=cv2.VideoCapture(0)

pts = deque(maxlen=64)

Lower_blue = np.array([110,50,50])
Upper_blue = np.array([130,255,255])
while True:
	ret, img=cap.read()
	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	kernel=np.ones((5,5),np.uint8)
	mask=cv2.inRange(hsv,Lower_blue,Upper_blue)
	mask = cv2.erode(mask, kernel, iterations=1)
	mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
	#mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
	mask = cv2.dilate(mask, kernel, iterations=2)
	res=cv2.bitwise_and(img,img,mask=mask)
	cnts,heir=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]
	draw=cv2.bitwise_and(img,img,mask=mask)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY); 
	center = None
 
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		if radius > 50:
			cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv2.circle(img, center, 5, (0, 0, 255), -1)
		
	pts.appendleft(center)
	for i in range(1,len(pts)):
		if pts[i-1]is None or pts[i] is None:
			continue
		thick = int(np.sqrt(len(pts) / float(i + 1)) * 2.5)
		thickness_color = int(np.sqrt(len(pts) / float(i + 1)) * 2)
		cv2.line(gray, pts[i-1],pts[i],(0,0,0),thick)
		cv2.line(draw, pts[i-1],pts[i],(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)),thickness_color) #random colors
		



	cv2.imshow("Frame", img)
	cv2.imshow("gray", gray)
	cv2.imshow("draw",draw)
	
	
	k=cv2.waitKey(30) & 0xFF
	if k==32:
		break
# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()

