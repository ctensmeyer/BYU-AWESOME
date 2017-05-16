
import cv2
import sys

im = cv2.imread(sys.argv[1], 0)
cv2.imwrite(sys.argv[2], 255 - im)
