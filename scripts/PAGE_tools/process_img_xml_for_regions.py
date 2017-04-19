import sys
import parse_PAGE
import cv2
import line_extraction
import numpy as np
import os
import traceback
import matplotlib.pyplot as plt
from collections import defaultdict


def handle_single_image(xml_path, img_path, output_directory, use_baseline=True):


    with open(xml_path) as f:
        num_lines = sum(1 for line in f.readlines() if len(line.strip())>0)

    img = cv2.imread(img_path)
    region_data = np.zeros(img.shape[:2], np.uint8)

    if num_lines > 0:

        xml_data = parse_PAGE.readXMLFile(xml_path)

        if len(xml_data) > 1:
            raise Exception("Not handling this correctly")

        region_cnt = len(xml_data[0]['all_region_types'])
        for i, (region_type, regions) in enumerate(xml_data[0]['all_region_types'].iteritems()):
            for region in regions:
                region_mask = line_extraction.extract_region_mask(img, region['bounding_poly'])
                region_mask[region_mask==255] = 1

                ## USE THIS FOR MAIN CLASS PREDICTION ##
                # region_data[region_mask != 0] = int((float(i+1) / region_cnt)*255)
                # region_data[region_mask != 0] == i
                ########################################

                ## USE THIS FOR SUBREGIONS ##
                region_mask = region_mask*(2**i)
                region_data = np.bitwise_or(region_mask, region_data)

                for j, (sub_region_type, sub_regions) in enumerate(region['subregions'].iteritems()):
                    for sub_region in sub_regions:
                        sub_region_mask = line_extraction.extract_region_mask(img, sub_region['bounding_poly'])
                        sub_region_mask[sub_region_mask==255] = 1
                        sub_region_mask = sub_region_mask*(2**j)
                        region_data = np.bitwise_or(sub_region_mask, region_data)
                #############################

        plt.imshow(region_data, cmap='spectral')
        plt.show()
    else:
        print "WARNING: {} has no lines".format(xml_path)

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

def find_best_xml(list_of_files, filename):

    if len(list_of_files) <= 1:
        return list_of_files

    print "Selecting multiple options from:"

    line_cnts = []
    for xml_path in list_of_files:
        test_xml_path = os.path.join(xml_path, filename+".xml")
        print test_xml_path
        with open(test_xml_path) as f:
            num_lines = sum(1 for line in f.readlines() if len(line.strip())>0)
        line_cnts.append((num_lines, xml_path))
    line_cnts.sort(key=lambda x:x[0], reverse=True)
    print "Sorted by line count..."
    ret = [l[1] for l in line_cnts]
    return ret


def process_dir(xml_directory, img_directory, output_directory, use_baseline=True):
    xml_filename_to_fullpath = defaultdict(list)
    for root, sub_folders, files in os.walk(xml_directory):
        for f in files:
            if not f.endswith(".xml"):
                continue
            f = f[:-len(".xml")]
            if f in xml_filename_to_fullpath:
                print "Error: this assumes no repeating files names: {} xml".format(f)

            xml_filename_to_fullpath[f].append(root)

    png_filename_to_fullpath = {}
    image_ext = {}
    for root, sub_folders, files in os.walk(img_directory):
        for f in files:
            valid_image_extensions = ['.jpg', '.png', '.JPG', '.PNG', '.tif']
            # if not f.endswith(".jpg") and not f.endswith(".png"):
            if not any([f.endswith(v) for v in valid_image_extensions]):
                continue

            if f in png_filename_to_fullpath:
                print "Error: this assumes no repeating files names: {} img".format(f)

            extension = f[-len(".png"):]
            f = f[:-len(".png")]
            image_ext[f] = extension
            png_filename_to_fullpath[f] = root

    xml_not_imgs = set(xml_filename_to_fullpath.keys()) - set(png_filename_to_fullpath.keys())
    print "Files in XML but not Images", len(xml_not_imgs)
    if len(xml_not_imgs) > 0:
        print xml_not_imgs
    img_not_xml = set(png_filename_to_fullpath.keys()) - set(xml_filename_to_fullpath.keys())
    print "Files in Images but not XML", len(img_not_xml)
    if len(img_not_xml) > 0:
        print img_not_xml
    print ""
    to_process = set(xml_filename_to_fullpath.keys()) & set(png_filename_to_fullpath.keys())
    print "Number to be processed", len(to_process)

    for i, filename in enumerate(to_process):
        if i%10==0:
            print i
        img_path = png_filename_to_fullpath[filename]
        xml_paths = xml_filename_to_fullpath[filename]

        out_rel = os.path.relpath(img_path, img_directory)
        this_output_directory = os.path.join(output_directory, out_rel)

        img_path = os.path.join(img_path, filename+image_ext[filename])
        success = False
        for xml_path in find_best_xml(xml_paths, filename):
            xml_path = os.path.join(xml_path, filename+".xml")
            try:
                handle_single_image(xml_path, img_path, this_output_directory, use_baseline)
                success = True
                break
            except KeyboardInterrupt:
                raise
            except Exception, e:
                print e
                print "Failed Attempt... ", filename
                continue

        if not success:
            out_str = filename+" Failed"
            print "".join(["*"]*len(out_str))
            print filename, "Failed"
            print "".join(["*"]*len(out_str))


if __name__ == "__main__":
    xml_directory = sys.argv[1]
    img_directory = sys.argv[2]
    output_directory = sys.argv[3]
    region_setting = sys.argv[4] if len(sys.argv) > 4 else None

    use_baseline = True
    if region_setting is not None:
        use_baseline = False

    process_dir(xml_directory, img_directory, output_directory, use_baseline=use_baseline)
