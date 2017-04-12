import sys
import os
import shutil
import subprocess

TMP_DIR = "results/tmp_xmls"

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

if __name__ == "__main__":
    xml_directory = sys.argv[1]
    txt_directory = sys.argv[2]
    list_file = sys.argv[3]

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    shutil.copytree(xml_directory, TMP_DIR)
    xml_filename_to_fullpath = get_xml_map(TMP_DIR)
    full_list = []
    for i, filename in enumerate(xml_filename_to_fullpath.keys()):
        root_path = xml_filename_to_fullpath[filename]

        out_rel = os.path.relpath(root_path, TMP_DIR)
        this_output_directory = os.path.join(txt_directory, out_rel)

        xml_path = os.path.join(root_path, filename+".xml")
        txt_path = os.path.join(this_output_directory, filename+".txt")
        try:
            baselines = apply_xml_convert(xml_path)
            if len(baselines) == 0:
                print "Skipping - No baselines: {}".format(xml_path)
                continue
            save_baselines(baselines, txt_path)
            full_list.append(txt_path)
        except KeyboardInterrupt:
            raise
        except Exception, e:
            print e
            out_str = filename+" Failed"
            print "".join(["*"]*len(out_str))
            print filename, "Failed"
            print "".join(["*"]*len(out_str))

    with open(list_file, 'w') as f:
        for l in full_list:
            f.write(l+"\n")
