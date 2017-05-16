
import sys
import cv2

cv2.imwrite(sys.argv[1], 255 * cv2.imread(sys.argv[1], -1))
