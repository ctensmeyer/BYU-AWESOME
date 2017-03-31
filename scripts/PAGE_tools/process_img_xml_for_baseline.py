import sys
import parse_PAGE
import cv2
import line_extraction
import numpy as np
import os
# import matplotlib.pyplot as plt


def handle_single_image(xml_path, img_path, output_directory):

    xml_data = parse_PAGE.readXMLFile(xml_path)
    img = cv2.imread(img_path)

    if len(xml_data) > 1:
        raise Exception("Not handling this correctly")

    region_data = np.zeros(img.shape[:2], np.uint8)

    for region in xml_data[0]['regions']:

        region_mask = line_extraction.extract_region_mask(img, region['bounding_poly'])

        for i, line in enumerate(xml_data[0]['lines']):
            if line['region_id'] != region['id']:
                continue
            line_mask = line_extraction.extract_baseline(img, line['baseline'])

            region_data[line_mask != 0] = 255

    basename = os.path.basename(xml_path)
    output_name = basename[:-len(".xml")]
    output_name += ".png"

    output_file = os.path.join(output_directory, output_name)
    if not os.path.exists(os.path.dirname(output_file)):
        try:
            os.makedirs(os.path.dirname(output_file))
        except OSError as exc:
            raise Exception("Could not write file")

    cv2.imwrite(output_file, region_data)

if __name__ == "__main__":
    xml_directory = sys.argv[1]
    img_directory = sys.argv[2]
    output_directory = sys.argv[3]

    xml_filename_to_fullpath = {}
    for root, sub_folders, files in os.walk(xml_directory):
        for f in files:
            if not f.endswith(".xml"):
                continue
            f = f[:-len(".xml")]
            if f in xml_filename_to_fullpath:
                print "Error: this assumes no repeating files names: {} xml".format(f)

            xml_filename_to_fullpath[f] = root

    png_filename_to_fullpath = {}
    image_ext = {}
    for root, sub_folders, files in os.walk(img_directory):
        for f in files:
            if not f.endswith(".jpg") and not f.endswith(".png"):
                continue

            if f in png_filename_to_fullpath:
                print "Error: this assumes no repeating files names: {} img".format(f)

            extension = f[-len(".png"):]
            f = f[:-len(".png")]
            image_ext[f] = extension
            png_filename_to_fullpath[f] = root

    print "Files in XML but not Images", len(set(xml_filename_to_fullpath.keys()) - set(png_filename_to_fullpath.keys()))
    print "Files in Images but not XML", len(set(png_filename_to_fullpath.keys()) - set(xml_filename_to_fullpath.keys()))
    print ""
    to_process = set(xml_filename_to_fullpath.keys()) & set(png_filename_to_fullpath.keys())
    print "Number to be processed", len(to_process)

    for i, filename in enumerate(to_process):
        if i%10==0:
            print i
        img_path = png_filename_to_fullpath[filename]
        xml_path = xml_filename_to_fullpath[filename]

        out_rel = os.path.relpath(img_path, img_directory)
        this_output_directory = os.path.join(output_directory, out_rel)

        xml_path = os.path.join(xml_path, filename+".xml")
        img_path = os.path.join(img_path, filename+image_ext[filename])
        try:
            handle_single_image(xml_path, img_path, this_output_directory)
        except KeyboardInterrupt:
            raise
        except:
            out_str = filename+" Failed"
            print "".join(["*"]*len(out_str))
            print filename, "Failed"
            print "".join(["*"]*len(out_str))
