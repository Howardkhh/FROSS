#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_path> <vis_folder>"
    exit 1
fi

dataset_path=$1
vis_folder=$2

scenes=("office_2" "office_3" "office_4" "room_0" "room_1" "room_2")
for scene in "${scenes[@]}"; do
    python3 visualize2D.py --scene $scene --dataset_path $dataset_path --vis_folder $vis_folder &
done

scenes=("office_2" "office_3" "office_4" "room_0" "room_1" "room_2")
for scene in "${scenes[@]}"; do
    python3 visualize3D.py --scene $scene --dataset_path $dataset_path --vis_folder $vis_folder &
done
wait

scenes=("office_2" "office_3" "office_4" "room_0" "room_1" "room_2")
for scene in "${scenes[@]}"; do
    python3 visualize3D_texts.py --scene $scene --dataset_path $dataset_path --vis_folder $vis_folder &
done
wait

scenes=("office_2" "office_3" "office_4" "room_0" "room_1" "room_2")
for scene in "${scenes[@]}"; do
    ffmpeg -framerate 15 -i "$vis_folder/2D/$scene/frame-%06d.color.png" -framerate 15 -i "$vis_folder/3D_text/$scene/frame-%06d.color.png" -filter_complex "[1:v][0:v]scale2ref=iw:iw*ih/iw[rb][ra];[ra][rb]vstack=inputs=2[v]" -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 24 -preset veryfast -shortest ${vis_folder}/${scene}.mp4
done
wait