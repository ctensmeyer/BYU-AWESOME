import numpy as np
import argparse
import glob
import cv2
import sys
import matplotlib.pyplot as plt
import math
from scipy import signal as sig

def hough_line(img, ori, org, required_votes):


    h,w = img.shape[:2]
    mid_pt = (np.ceil(w/2.0), np.ceil(h/2.0)) #consider odd pixel w/h
    max_diagonal = np.ceil(np.sqrt(mid_pt[0]**2 + mid_pt[1]**2))

    accum_size = max_diagonal
    # accum_size = 100
    rhos = np.linspace(-max_diagonal, max_diagonal, accum_size)
    # thetas = np.linspace(-np.pi/2.0, np.pi/2.0, 360)
    thetas = np.linspace(-np.pi/2.0, np.pi/2.0, accum_size)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.float64)
    accumulator2 = np.zeros((len(rhos), len(thetas)), dtype=np.float64)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    y_idxs, x_idxs = np.nonzero(img)
    angles = []
    i=0
    for x, y in zip(x_idxs, y_idxs):
        if i%100 == 0:
            print i, "/", len(x_idxs)
        i+=1
        angle = ori[y,x]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        for theta_idx, theta in enumerate(thetas):

            v1 = np.array([cos_a, sin_a])
            v2 = np.array([cos_t[theta_idx], sin_t[theta_idx]])
            out = abs(np.dot(v1, v2))
            if out < 0.9:
                continue

            # scaler = 1.0
            scaler = 1.0 - (1.0 - out) / (1.0 - 0.9)


            rho = x * cos_t[theta_idx] + y * sin_t[theta_idx]
            rho = (x - mid_pt[0]) * cos_t[theta_idx] + (y - mid_pt[1]) * sin_t[theta_idx] + max_diagonal
            rho_idx = rho * accum_size / float(2.0 * max_diagonal)
            frac, whole = math.modf(rho_idx)
            # accumulator2[int(whole), theta_idx] += 1.0
            accumulator2[int(whole), theta_idx] += (1.0 - frac) * scaler
            next_whole = (int(whole)+1) % accum_size
            accumulator2[int(next_whole), theta_idx] += frac * scaler

    plt.imsave('accum.png', accumulator2, cmap='gray')

    np.save("acc.npy", accumulator2)

    print np.amax(accumulator2)
    print np.amax(accumulator)

    accumulator2 = np.load("acc.npy")

    accumulator2[accumulator2 < required_votes] = 0
    accumulator2_blurred = sig.convolve2d(accumulator2, np.ones((5,5)), mode='same') / 9.0

    plt.imsave('accum_thres.png', accumulator2, cmap='gray')
    plt.imsave('accum_blurred.png', accumulator2, cmap='gray')

    # plt.imshow(accumulator2)
    # plt.show()

    i = 0
    while np.amax(accumulator2) > 0:
        i += 1
        print i

        best = np.argmax(accumulator2)
        best = np.unravel_index(best, accumulator2.shape)
        rho = rhos[best[0]]
        theta = thetas[best[1]]

        f_sin = np.sin(theta)
        f_cos = np.cos(theta)

        if abs(f_sin) < abs(f_cos):
            y1 = 0.0
            y2 = h

            x1 = (rho - (y1 - mid_pt[1]) * f_sin) / f_cos + mid_pt[0]
            x2 = (rho - (y2 - mid_pt[1]) * f_sin) / f_cos + mid_pt[0]
        else:
            x1 = 0.0
            x2 = w

            y1 = (rho - (x1 - mid_pt[0]) * f_cos) / f_sin + mid_pt[1]
            y2 = (rho - (x2 - mid_pt[0]) * f_cos) / f_sin + mid_pt[1]

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)

        X = accumulator2.shape[0]
        Y = accumulator2.shape[1]
        neighbors = lambda x, y : [(x2, y2) for x2 in range(x-1, x+2)
                               for y2 in range(y-1, y+2)
                               if (-1 < x < X and
                                   -1 < y < Y and
                                   (x != x2 or y != y2) and
                                   (0 <= x2 < X) and
                                   (0 <= y2 < Y))]

        cv2.line(org,(x1,y1),(x2,y2),(255,0,0),3)

        to_visit = [best]
        while len(to_visit) > 0:
            cur = to_visit.pop(0)
            if accumulator2_blurred[cur[0], cur[1]] == 0:
                continue
            accumulator2_blurred[cur[0], cur[1]] = 0
            accumulator2[cur[0], cur[1]] = 0
            to_add = [v for v in neighbors(cur[0], cur[1])]
            to_add = [np.array(v) for v in to_add]
            add_all = []
            for v in to_add:
                if accumulator2_blurred[v[0], v[1]] != 0:
                    add_all.append(v)
            to_visit.extend(add_all)

    return org
    # plt.imshow(org, cmap='gray')
    # plt.show()

    # plt.imshow(accumulator2, cmap='gray')
    # plt.show()

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)


	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	# return the edged image
	return edged


def main():
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    out_path = sys.argv[2]
    org = img

    #
    # img = cv2.medianBlur(img,5)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY,11,2)
    #
    # img = 255 - img

    img = cv2.medianBlur(img,5)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = auto_canny(img)
    plt.imsave('edge_map.png', out, cmap='gray')
    # plt.imshow(out, cmap='gray', interpolation='none')
    # plt.show()

    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    ori = np.arctan2(sobelx, sobely)

    ori = -ori % np.pi - np.pi/2.0
    final = hough_line(out, ori, org, 65)

    cv2.imwrite(out_path, final)

if __name__ == "__main__":
    main()
