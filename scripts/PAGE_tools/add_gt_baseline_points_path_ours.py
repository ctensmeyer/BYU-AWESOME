import sys
import os
import parse_PAGE

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
    output_txt_folder_path = sys.argv[3]

    with open(input_paths_path) as f:
        input_paths = json.load(f)
    output_paths = deepcopy(input_paths)

    for i, input_path in enumerate(input_paths):

        global_input_path = input_path['gt_xml_path']

        base = os.path.basename(global_input_path)
        filename = os.path.splitext(base)[0]

        output_filename = os.path.join(output_txt_folder_path, "{}-{}.txt".format(filename, i))

        baselines = []
        xml_data = parse_PAGE.readXMLFile(global_input_path)
        if len(xml_data) > 1:
            raise Exception("Not handling this correctly")

        for j, line in enumerate(xml_data[0]['lines']):
            baseline = line['baseline']
            baselines.append(baseline)

        str_baselines = []
        for b in baselines:
            b = [",".join([str(x) for x in a]) for a in b]
            str_baselines.append(";".join(b))

        save_baselines(str_baselines, output_filename)

        output_paths[i]['gt_baseline_points_path'] = output_filename

        # print output_filename

    with open(output_paths_path, 'w') as f:
        json.dump(output_paths, f)
