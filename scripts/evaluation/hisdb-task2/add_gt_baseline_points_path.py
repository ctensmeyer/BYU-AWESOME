import sys
import os
import shutil
import subprocess

def get_xml_map(xml_directory):
    xml_filename_to_fullpath = {}
    for root, sub_folders, files in os.walk(xml_directory):
        for f in files:
            if not f.endswith(".xml"):
                continue
            f = f[:-len(".xml")]
            if f in xml_filename_to_fullpath:
                print "Error: this assumes no repeating files names: {} xml".format(f)

            xml_filename_to_fullpath[f] = root
    return xml_filename_to_fullpath

def apply_xml_convert(filepath):
    res = subprocess.call(['java', '-jar', 'built_jars/convert_xml.jar', filepath])
    raw_input()

    if res != 0:
        raise Exception("Java problem")

    dirname = os.path.dirname(filepath)
    output_baselines = []
    cnt = 0
    for f in os.listdir(dirname):
        if f.endswith(".txt"):
            if cnt > 0:
                print "WARNING: More than one region, this could mean a delete failed"
            with open(os.path.join(dirname, f)) as read_f:
                lines = read_f.readlines()
            lines = lines[1:]
            lines = [l.strip() for l in lines if len(l.strip()) > 0]
            output_baselines.extend(lines)
            os.remove(os.path.join(dirname, f))
            cnt += 1

    return output_baselines

def save_baselines(baselines, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        try:
            os.makedirs(os.path.dirname(output_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    with open(output_path, 'w') as f:
        for b in baselines:
            f.write(b+"\n")

import sys
import json
from copy import deepcopy

if __name__ == "__main__":
    input_paths_path = sys.argv[1]
    output_paths_path = sys.argv[2]
    global_base_path = sys.argv[3]
    output_txt_folder_path = sys.argv[4]

    with open(input_paths_path) as f:
        input_paths = json.load(f)
    output_paths = deepcopy(input_paths)

    for i, input_path in enumerate(input_paths):
        global_input_path = os.path.join(global_base_path, input_path['gt_xml_path'])

        base = os.path.basename(global_input_path)
        filename = os.path.splitext(base)[0]

        output_filename = os.path.join(output_txt_folder_path, "{}-{}.txt".format(filename, i))


        baselines = apply_xml_convert(global_input_path)
        save_baselines(baselines, output_filename)

        save_path = os.path.relpath(output_filename, global_base_path)
        output_paths[i]['gt_baseline_points_path'] = save_path

    with open(output_paths_path, 'w') as f:
        json.dump(output_paths, f)
