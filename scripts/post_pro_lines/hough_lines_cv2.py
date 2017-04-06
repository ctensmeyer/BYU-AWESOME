import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = sys.argv[1]

img = cv2.imread(img_path)
img = 255-img #np.subtract( np.array([[[255,255,255]]]), img) #invert, white is on
#plt.imshow(img)
#plt.show()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#dilate
#kernel = np.ones((7,7),np.uint8)

#img = cv2.dilate(img, kernel, iterations = 1)
#gray = cv2.dilate(gray, kernel, iterations = 1)

img[:,:,0] = 0
img[:,:,1] = 0

line_img = np.zeros_like(img)
minLineLength = 200
maxLineGap = 600
lines = cv2.HoughLinesP(gray,1,np.pi/180,600,minLineLength,maxLineGap)
lines = lines.reshape(-1, 4)
for x1,y1,x2,y2 in lines:
    cv2.line(line_img,(x1,y1),(x2,y2),(0,255,0),2)

display_img = line_img + img
for x1,y1,x2,y2 in lines:
    cv2.circle(display_img, (x1,y1), 2, (255,0,0))
    cv2.circle(display_img, (x2,y2), 2, (255,0,0))

plt.imshow(display_img)
plt.show()
