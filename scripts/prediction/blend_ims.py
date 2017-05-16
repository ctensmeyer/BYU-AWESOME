
import cv2
import numpy as np
import sys

predictions = cv2.imread(sys.argv[1], 1)
im = cv2.imread(sys.argv[2], 1)
assert predictions.shape == im.shape

if len(sys.argv) >= 5 and sys.argv[4] == 'invert':
	mask = np.logical_or(np.logical_or(predictions[:,:,0] != 0, predictions[:,:,1] != 0),  predictions[:,:,2] != 0)
else:
	mask = np.logical_or(np.logical_or(predictions[:,:,0] != 255, predictions[:,:,1] != 255),  predictions[:,:,2] != 255)
mask = np.concatenate([mask[:,:,np.newaxis]] * 3, axis=2)

out = (0.8 * mask) * predictions + 0.2 * mask * im + (1 - mask) * im

cv2.imwrite(sys.argv[3], out)


