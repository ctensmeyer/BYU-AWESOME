
import sys
import cv2
import numpy as np

out_file = sys.argv[1]
ims = list()
for fn in sys.argv[2:]:
	im = cv2.imread(fn, 0)
	if im is None:
		print fn
	ims.append(im[np.newaxis,:,:])
ims = np.concatenate(ims, axis=0)
avg_im = ims.mean(axis=0, dtype=float)

high_indices = avg_im > 128
low_indices = avg_im <= 128
avg_im[high_indices] = 255
avg_im[low_indices] = 0

avg_im = avg_im.astype(np.uint8)
cv2.imwrite(out_file, avg_im)

