#!/usr/bin/env python3
#make sure opencv-python is installed locally
import numpy as np
import cv2
import argparse
from collections import deque
import random


cap=cv2.VideoCapture(0)

pts_red = deque(maxlen=64)
pts_blue = deque(maxlen=64)

Lower_blue = np.array([110, 50, 50]) # set lower bound for color recognition
Upper_blue = np.array([130, 255, 255]) # set upper bound for color recognition

# blue detection range actually ranges from dark blue to a light yellow - weird?

Lower_red = np.array([50, 50, 110]) # TRYING TO DETECT RED - COLOR INTERVAL WRONG
Upper_red = np.array([255, 255, 130]) # TRYING TO DETECT RED - COLOR INTERVAL WRONG

# tried to reverse RGB and BGR -- with this reversal ranges from dark red to light cyan-ish?
# if we don't get this figured out by tonight we should email bove and see what he makes of it?

lower = np.array([0,0,0]) # BLACK for black screen
upper = np.array([0,0,0]) # BLACK for black screen

while True:
	ret, img=cap.read()
	hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	kernel=np.ones((5,5),np.uint8)
	mask_blue=cv2.inRange(hsv,Lower_blue,Upper_blue)
	mask2=cv2.inRange(hsv, lower, upper)
	mask_blue=cv2.erode(mask_blue, kernel, iterations=1)
	mask_blue=cv2.morphologyEx(mask_blue,cv2.MORPH_OPEN,kernel)
	mask_blue = cv2.dilate(mask_blue, kernel, iterations=2)
	res=cv2.bitwise_and(img,img,mask=mask_blue)
	cnts, heir=cv2.findContours(mask_blue.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2:]

	mask_red = cv2.inRange(hsv, Lower_red, Upper_red)
	mask_red = cv2.erode(mask_red, kernel, iterations=1)
	mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
	mask_red = cv2.dilate(mask_red, kernel, iterations=2)
	res_red=cv2.bitwise_and(img, img, mask=mask_red)
	cnts_red, heir_red = cv2.findContours(mask_red.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]


	draw=cv2.bitwise_and(img, img, mask=mask2)
	gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	center = None

# DETECT BLUE
	if len(cnts) > 0:
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		if radius > 50: # can we make it smaller to see it on a larger frame?
			cv2.circle(img, (int(x), int(y)), int(radius),(0, 255, 255), 2)
			cv2.circle(img, center, 5, (0, 0, 255), -1)
		
	pts_blue.appendleft(center)
	for i in range(1,len(pts_blue)):
		if pts_blue[i-1]is None or pts_blue[i] is None:
			continue
		thick = int(np.sqrt(len(pts_blue) / float(i + 1)) * 2.5)
		thickness_color = int(np.sqrt(len(pts_blue) / float(i + 1)) * 2)
		cv2.line(gray, pts_blue[i-1],pts_blue[i],(0,0,0),thick)
		cv2.line(draw, pts_blue[i-1],pts_blue[i],(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)),thickness_color) #random colors

# DETECT RED
	if len(cnts_red) > 0:
		c = max(cnts_red, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

		if radius > 50:  # can we make it smaller to see it on a larger frame?
			cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2)
			cv2.circle(img, center, 5, (0, 0, 255), -1)

	pts_red.appendleft(center)
	for i in range(1, len(pts_red)):
		if pts_red[i - 1] is None or pts_red[i] is None:
			continue
		thick = int(np.sqrt(len(pts_red) / float(i + 1)) * 2.5)
		thickness_color = int(np.sqrt(len(pts_red) / float(i + 1)) * 2)
		cv2.line(gray, pts_red[i - 1], pts_red[i], (0, 0, 0), thick)
		cv2.line(draw, pts_red[i - 1], pts_red[i], (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
				 thickness_color)  # random colors

	cv2.imshow("frame", img) # color image
	cv2.imshow("gray", gray) # grayscale image w/ black pen
	cv2.imshow("draw", draw) # black screen with color drawing


	k=cv2.waitKey(100) & 0xFF # CHANGED TO 100: makes wait/flashing lights slower
	if k==32:
		break

# cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()

