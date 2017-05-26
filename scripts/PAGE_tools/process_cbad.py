import sys
import os
import process_img_xml_for_baseline
import json

if __name__ == "__main__":
    root_path = sys.argv[1]
    output_directory = sys.argv[2]
    results_paths_path = sys.argv[3]
    sub_dirs = next(os.walk(root_path))[1]

    all_results_path = []
    for d in sub_dirs:

        print "Processing Folder: {}".format(d)
        dir_to_process = os.path.join(root_path, d)
        dir_to_output = os.path.join(output_directory, d)
        result_paths = process_img_xml_for_baseline.process_dir(dir_to_process, dir_to_process, dir_to_output)
        all_results_path.extend(result_paths)

    with open(results_paths_path, 'w') as f:
        json.dump(all_results_path, f)
