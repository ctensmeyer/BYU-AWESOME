import sys
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json
from copy import deepcopy
import time

def pred_to_pts(color_img):
    global_threshold = 127
    slice_size = 25
    # small_threshold = 0
    small_threshold = 250
    img = cv2.cvtColor( color_img, cv2.COLOR_RGB2GRAY )
    ret, th = cv2.threshold(img,global_threshold,255,cv2.THRESH_BINARY)
    connectivity = 4
    s = time.time()
    output= cv2.connectedComponentsWithStats(th, connectivity, cv2.CV_32S)
    baselines = []
    #skip background
    for label_id in xrange(1, output[0]):
        min_x = output[2][label_id][0]
        min_y = output[2][label_id][1]
        max_x = output[2][label_id][2] + min_x
        max_y = output[2][label_id][3] + min_y
        cnt = output[2][label_id][4]

        if cnt < small_threshold:
            continue

        baseline = output[1][min_y:max_y, min_x:max_x]

        pts = []
        x_all, y_all = np.where(baseline == label_id)
        first_idx = y_all.argmin()
        first = (y_all[first_idx]+min_x, x_all[first_idx]+min_y)

        pts.append(first)
        for i in xrange(0, baseline.shape[1], slice_size):
            next_i = i+slice_size
            baseline_slice = baseline[:, i:next_i]

            x, y = np.where(baseline_slice == label_id)
            x_avg = x.mean()
            y_avg = y.mean()
            pts.append((int(y_avg+i+min_x), int(x_avg+min_y)))

        last_idx = y_all.argmax()
        last = (y_all[last_idx]+min_x, x_all[last_idx]+min_y)
        pts.append(last)

        if len(pts) <= 1:
            continue

        baselines.append(pts)

    # img_copy = color_img.copy()
    # for b in baselines:
    #     pts = np.array(b, np.int32)
    #     pts = pts.reshape((-1,1,2))
    #     cv2.polylines(img_copy,[pts],False,(0,255,255), thickness=1)
    #
    # plt.imshow(img_copy)
    # plt.show()

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
    cnt = 0
    for i, input_path in enumerate(input_paths):
        if i%10 == 0:
            print i, cnt, (time.time() - start) / (i+1)
        # image_path = input_path['gt_pixel_img_path']
        image_path = input_path['pred_pixel_path']

        save_path = os.path.basename(image_path)
        save_path = os.path.splitext(save_path)[0]
        save_path = "{}-{}.txt".format(save_path, i)
        save_path = os.path.join(output_txt_folder_path, save_path)

        img = cv2.imread(image_path)
        baselines = pred_to_pts(img)
        cnt += len(baselines)
        write_baseline_pts(baselines, save_path)

        output_path = deepcopy(input_path)
        output_path['pred_baseline_path'] = save_path
        output_paths.append(output_path)


    with open(output_paths_path, 'w') as f:
        json.dump(output_paths, f)
