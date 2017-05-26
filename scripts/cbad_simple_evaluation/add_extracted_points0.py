import sys
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from copy import deepcopy
import time

TIME_IN_CC = 0
TIME_IN_UNI = 0
TIME_IN_SPLIT = 0

def pred_to_pts(color_img):
    global TIME_IN_CC, TIME_IN_UNI, TIME_IN_SPLIT
    global_threshold = 127
    slice_size = 25
    small_threshold = 500


    img = cv2.cvtColor( color_img, cv2.COLOR_RGB2GRAY )
    ret, th = cv2.threshold(img,global_threshold,255,cv2.THRESH_BINARY)
    connectivity = 4
    s = time.time()
    output = cv2.connectedComponentsWithStats(th, connectivity, cv2.CV_32S)[1]
    TIME_IN_CC += (time.time() - s)

    s = time.time()
    unique_ids, unique_counts = np.unique(output, return_counts=True)
    TIME_IN_UNI += (time.time() - s)

    s = time.time()
    baselines = []
    for unique_id, unique_counts in zip(unique_ids, unique_counts):
        if unique_id == 0:
            #Skip background
            continue

        if unique_counts < small_threshold:
            continue
        baseline = np.zeros_like(th)
        baseline[output==unique_id] = 255

        # region_pixels = np.where(output==unique_id)
        # max_y = max(region_pixels[0])
        # min_y = min(region_pixels[0])
        #
        # max_x = max(region_pixels[1])
        # min_x = min(region_pixels[1])
        #
        # sub_region = img[min_y:max_y, min_x:max_x]

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
    TIME_IN_SPLIT += (time.time() - s)

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
    input_paths_path = sys.argv[1]
    output_paths_path = sys.argv[2]
    output_txt_folder_path = sys.argv[3]

    with open(input_paths_path) as f:
        input_paths = json.load(f)


    output_paths = []
    start = time.time()
    for i, input_path in enumerate(input_paths):
        if i%10 == 0:
            print i, (time.time() - start) / (i+1)
            print "CC", TIME_IN_CC / (i+1)
            print "UN", TIME_IN_UNI / (i+1)
            print "SP", TIME_IN_SPLIT / (i+1)
        image_path = input_path['gt_pixel_img_path']
        # image_path = input_path['pred_pixel_path']

        save_path = os.path.basename(image_path)
        save_path = os.path.splitext(save_path)[0]
        save_path = "{}-{}.txt".format(save_path, i)
        save_path = os.path.join(output_txt_folder_path, save_path)

        img = cv2.imread(image_path)
        baselines = pred_to_pts(img)
        write_baseline_pts(baselines, save_path)

        output_path = deepcopy(input_path)
        output_path['pred_baseline_path'] = save_path
        output_paths.append(output_path)

    with open(output_paths_path, 'w') as f:
        json.dump(output_paths, f)
