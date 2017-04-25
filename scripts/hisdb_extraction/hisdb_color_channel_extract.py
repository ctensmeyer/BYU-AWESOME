import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

images_folder = sys.argv[1]
output_folder = sys.argv[2]

files = next(os.walk(images_folder))[2]
_______ = [0, 1, 2, 3, 4, 5,  6,  7,  8,  9]
options = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16]
mappings = {}
for i, o in enumerate(options):
    mappings[o] = i

for i, f in enumerate(files):
    if i%10 == 0:
        print i
    img = cv2.imread(os.path.join(images_folder, f))

    b_channel, g_channel, r_channel = cv2.split(img)

    b_channel[r_channel==128] = 16

    vals =  np.unique(b_channel)
    for v in vals:
        if v not in options:
            print "Previous not considered type"
            print v

    # raw_input()

    background_ch = np.bitwise_and(b_channel, np.full_like(b_channel, 1))
    comment_ch = np.bitwise_and(b_channel, np.full_like(b_channel, 2))
    decoration_ch = np.bitwise_and(b_channel, np.full_like(b_channel, 4))
    main_text_body_ch = np.bitwise_and(b_channel, np.full_like(b_channel, 8))
    boundary_ch = np.bitwise_and(r_channel, np.full_like(r_channel, 128))



    background_ch[background_ch!=0] = 255
    main_text_body_ch[main_text_body_ch!=0] = 255
    decoration_ch[decoration_ch!=0] = 255
    comment_ch[comment_ch!=0] = 255
    boundary_ch[boundary_ch!=0] = 255

    all_labels = np.zeros_like(boundary_ch)
    for i, o in enumerate(options):
        all_labels[b_channel==o] = i




    f = f[:-len(".png")]

    img_output_folder = os.path.join(output_folder, f)

    if not os.path.exists(img_output_folder):
        try:
            os.makedirs(img_output_folder)
        except OSError as exc:
            raise Exception("Could not write file")

    cv2.imwrite(os.path.join(img_output_folder, "background.png"), background_ch)
    cv2.imwrite(os.path.join(img_output_folder, "main_text_body.png"), main_text_body_ch)
    cv2.imwrite(os.path.join(img_output_folder, "decoration.png"), decoration_ch)
    cv2.imwrite(os.path.join(img_output_folder, "comment.png"), comment_ch)
    cv2.imwrite(os.path.join(img_output_folder, "boundary.png"), boundary_ch)
    cv2.imwrite(os.path.join(img_output_folder, "all_labels.png"), all_labels)
