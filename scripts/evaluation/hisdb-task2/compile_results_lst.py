import sys
import json
import os

def write_lst_file(output_path, data):
    with open(output_path, 'w') as f:
        for d in data:
            f.write(d+"\n")

if __name__ == "__main__":
    input_paths_path = sys.argv[1]
    train_type = sys.argv[2]
    output_pred_path = sys.argv[3]
    output_gt_path = sys.argv[4]
    with open(input_paths_path) as f:
        input_paths = json.load(f)

    input_paths = [ip for ip in input_paths if ip['train_type'] == train_type]

    print len(input_paths)

    preds = []
    gts = []

    for ip in input_paths:
        preds.append(os.path.abspath(ip['pred_baseline_path']))
        gts.append(os.path.abspath(ip['gt_baseline_points_path']))


    write_lst_file(output_pred_path, preds)
    write_lst_file(output_gt_path, gts)
