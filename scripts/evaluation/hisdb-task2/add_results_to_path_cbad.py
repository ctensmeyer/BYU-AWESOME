import sys
import json
from copy import deepcopy
import os

if __name__ == "__main__":
    results_folder = sys.argv[1]
    input_paths_path = sys.argv[2]
    output_paths_path = sys.argv[3]

    with open(input_paths_path) as f:
        input_paths = json.load(f)

    # output_paths = deepcopy(input_paths)
    output_paths = []

    results_images = {}
    for train_type in ['train', 'val', 'test']:
        tmp_results_folder = os.path.join(results_folder, train_type, 'raw')

        for root, dirs, files in os.walk(tmp_results_folder):
            for name in files:
                compare_name = os.path.splitext(name)[0]
                results_images[compare_name] = (os.path.join(root, name), train_type)

    # for r in results_images:
    parsed_input_paths = {}
    for input_path in input_paths:
        compare_path = input_path['gt_pixel_img_path']
        compare_path = os.path.normpath(compare_path)
        compare_path = os.path.join(*compare_path.split("/")[-2:])
        compare_path = compare_path.replace("/","_")
        compare_path = os.path.splitext(compare_path)[0]

        parsed_input_paths[compare_path] = input_path

    print len(results_images)
    print len(input_paths)
    print len(parsed_input_paths)

    for k, v in results_images.iteritems():
        if k not in parsed_input_paths:
            print "Error:", k

        new_paths = deepcopy(parsed_input_paths[k])
        filename, train_type = v
        new_paths['pred_pixel_path'] = filename
        new_paths['train_type'] = train_type

        output_paths.append(new_paths)

    with open(output_paths_path, 'w') as f:
        json.dump(output_paths, f)
