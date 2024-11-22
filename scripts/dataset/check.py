import os
from tqdm import tqdm
import pathlib
import argparse

parser = argparse.ArgumentParser(description='Check 3RScan dataset.')
parser.add_argument('--path', type=pathlib.Path, required=True, help='3RScan directory to check integrity')
args = parser.parse_args()

data_folder = args.path / "data"
num_folder = len(os.listdir(data_folder))

seq_folder_exist = 0
all_img_exist = 0
num_img = 0
num_bb = 0

for dir in tqdm(sorted(os.listdir(data_folder))):
    if os.path.isdir(os.path.join(data_folder, dir, "sequence")):
        if not (os.path.isfile(os.path.join(data_folder, dir, "sequence", "_info.txt"))):
            print(f"{dir} does not have _info.txt.")
            continue
        seq_folder_exist += 1
        with open(os.path.join(data_folder, dir, "sequence", "_info.txt"), 'r') as f:
            for line in f.readlines():
                if line.startswith("m_frames.size"):
                    n = int(line.split(" = ")[1])
        if len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith("color.jpg") and not f.endswith("rendered.color.jpg")]) == n:
            all_img_exist += 1
            num_img += n
        else:
            print(f"{dir} does not have all images. {n} images expected, {len([f for f in os.listdir(os.path.join(data_folder, dir, 'sequence')) if f.endswith('color.jpg') and not f.endswith('rendered.color.jpg')])} images found.")
        if len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith("bb.txt")]) == n:
            num_bb += n

print(f"Number of folders: {num_folder}")
print(f"Number of folders with sequence folder: {seq_folder_exist}")
print(f"Number of folders with all images: {all_img_exist}")
print(f"Number of images: {num_img}")
print(f"Number of images with bounding box files: {num_bb}")
