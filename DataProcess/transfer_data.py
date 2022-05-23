import glob
import os
import json
import shutil

img_path = glob.glob("/Data/val/*.JPEG")
target_path = "/Data/data_for_pretrain/val/"

with open("/Data/val/val.txt", "r") as f:
    lines = f.readlines()
    label_metas = {}
    for line in lines:
        info = json.loads(line)
        label_metas.update({info['filename']:info})

for img in img_path:
    label = str(label_metas[os.path.basename(img)]["label"])
    os.makedirs(os.path.join(target_path, label), exist_ok=True)
    shutil.copy(img, os.path.join(target_path, label, os.path.basename(img)))

