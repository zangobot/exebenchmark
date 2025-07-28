import os
from pathlib import Path

root_path = Path("/Users/zangobot/Documents/Datasets/Windows executables")

malware_folder = root_path / "malware"
goodware_folder = root_path / "goodware"

all_malware = sorted(malware_folder.glob("*"))
all_goodware = sorted(goodware_folder.glob("*"))

all_malware = [[m, 1] for m in all_malware]
all_goodware = [[m, 0] for m in all_goodware]

len_all_malw = len(all_malware)
len_all_good = len(all_goodware)


def dump_samples_in_file(malware_list, goodware_list, filepath):
    with open(filepath, "w") as f:
        for mlw, label in malware_list:
            if "DS_STORE" in str(mlw):
                continue
            f.write(f"{str(mlw)},{label}\n")
        for gdw, label in goodware_list:
            if "DS_STORE" in str(gdw):
                continue
            f.write(f"{str(gdw)},{label}\n")


dump_samples_in_file(all_malware[:20], all_goodware[:20], "training.txt")
dump_samples_in_file(all_malware[20: 30],all_goodware[20: 30], "test.txt")
dump_samples_in_file(all_malware[30: 40], all_goodware[30:40], "validation.txt")
