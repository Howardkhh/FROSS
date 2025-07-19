import os
from tqdm import tqdm
import pathlib
import argparse

parser = argparse.ArgumentParser(description='Check 3RScan dataset.')
parser.add_argument('--path', type=pathlib.Path, required=True, help='3RScan directory to check integrity')
args = parser.parse_args()

data_folder = args.path / "data"
scenes = [f for f in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, f))]
num_folder = len(scenes)

seq_folder_exist = 0
all_img_exist = 0
num_img = 0
num_bb = 0
num_rendered_color = 0
num_rendered_depth = 0
num_label = 0
num_visibility = 0
num_instance = 0

for dir in tqdm(sorted(scenes)):
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
        elif dir == "7ab2a9cb-ebc6-2056-8ba2-7835e43d47d3" and n == 62 and len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith("color.jpg") and not f.endswith("rendered.color.jpg")]) == 323:
            # This is a special case where the number of images is not consistent with the _info.txt file in the 3RScan dataset.
            all_img_exist += 1
            num_img += 323
        else:
            print(f"{dir} does not have all images. {n} images expected, {len([f for f in os.listdir(os.path.join(data_folder, dir, 'sequence')) if f.endswith('color.jpg') and not f.endswith('rendered.color.jpg')])} images found.")
            num_img += len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith("color.jpg") and not f.endswith("rendered.color.jpg")])
        if len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith("bb.txt")]) == n:
            num_bb += n
        else:
            num_bb += len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith("bb.txt")])
        if len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith(".rendered.color.jpg")]) == n:
            num_rendered_color += n
        else:
            num_rendered_color += len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith(".rendered.color.jpg")])
        if len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith(".rendered.depth.png")]) == n:
            num_rendered_depth += n
        else:
            num_rendered_depth += len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith(".rendered.depth.png")])
        if len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith(".rendered.labels.png")]) == n:
            num_label += n
        else:
            num_label += len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith(".rendered.labels.png")])
        if len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith(".visibility.txt")]) == n:
            num_visibility += n
        else:
            num_visibility += len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith(".visibility.txt")])
        if len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith("rendered.instances.png")]) == n:
            num_instance += n
        else:
            num_instance += len([f for f in os.listdir(os.path.join(data_folder, dir, "sequence")) if f.endswith("rendered.instances.png")])


print(f"Number of folders: {num_folder}")
print(f"Number of folders with sequence folder: {seq_folder_exist}")
print(f"Number of folders with all images: {all_img_exist}")
print(f"Number of images: {num_img}")
print(f"Number of images with bounding box files: {num_bb}")
print(f"Number of rendered color images: {num_rendered_color}")
print(f"Number of rendered depth images: {num_rendered_depth}")
print(f"Number of rendered label images: {num_label}")
print(f"Number of visibility files: {num_visibility}")
print(f"Number of instance files: {num_instance}")
if num_img != num_bb or num_img != num_rendered_color or num_img != num_rendered_depth or num_img != num_label or num_img != num_visibility or num_img != num_instance:
    print("There are inconsistencies in the dataset. Please check the output above.")
    print("Please note that the rio_renderer would exit peacefully in headless mode, but it would not render the images.")
    print("If the rendered images are missing, please make sure the extract_and_preprocess_3RScan.py script is run in non-headless mode.")
