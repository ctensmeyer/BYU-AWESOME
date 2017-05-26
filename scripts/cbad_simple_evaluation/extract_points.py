import sys
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def pred_to_pts(color_img):

    global_threshold = 127
    slice_size = 25
    small_threshold = 500


    img = cv2.cvtColor( color_img, cv2.COLOR_RGB2GRAY )
    ret, th = cv2.threshold(img,global_threshold,255,cv2.THRESH_BINARY)
    connectivity = 4
    output = cv2.connectedComponentsWithStats(th, connectivity, cv2.CV_32S)[1]

    unique_ids, unique_counts = np.unique(output, return_counts=True)
    baselines = []
    for unique_id, unique_counts in zip(unique_ids, unique_counts):
        if unique_id == 0:
            #Skip background
            continue

        baseline = np.zeros_like(th)
        baseline[output==unique_id] = 255

        if baseline.sum() / 255 < small_threshold:
            continue

        pts = []
        for i in xrange(0, baseline.shape[1], slice_size):
            next_i = i+slice_size
            baseline_slice = baseline[:, i:next_i]
            if baseline_slice.sum() == 0:
                continue
            x, y = np.where(baseline_slice != 0)
            x_avg = x.mean()
            y_avg = y.mean()
            pts.append((int(y_avg+i), int(x_avg)))

        baselines.append(pts)

    return baselines

def write_baseline_pts(baselines, filename):
    with open(filename, 'w') as f:
        for baseline in baselines:
            baseline_txt = []
            for pt in baseline:
                pt_txt = "{},{}".format(*pt)
                baseline_txt.append(pt_txt)
            f.write(";".join(baseline_txt)+"\n")

if __name__ == "__main__":
    image_path = sys.argv[1]
    output_path = sys.argv[2]

    img = cv2.imread(image_path)
    baselines = pred_to_pts(img)
    write_baseline_pts(baselines, output_path)
