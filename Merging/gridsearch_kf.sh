#!/bin/bash
set -e

split=test
output_path=output/slam_ablation

# trans_thresh=(0.0 0.01 0.02 0.03)
# rot_thresh=(999.0) # 1, 10, none degree

# i=0
# for rot in "${rot_thresh[@]}"; do
#     for trans in "${trans_thresh[@]}"; do
#         ((i=i+1))
#         (
#             # python main.py --dataset_path ../Datasets/Replica/ --artifact_path ../weights/RT-DETR-EGTR/VG/egtr__RT-DETR__VG__last.pth/batch__6__epochs__50_25__lr__2e-07_2e-06_2e-05__finetune/version_0/ --output_path ${output_path} --split ${split} --kf_strategy spatial --kf_translation ${trans} --kf_rotation ${rot} --label_categories replica

#             python evaluate.py --dataset_path ../Datasets/Replica/ --split ${split} --prediction_path ${output_path}/replica/predictions_gaussian_obj0.7_rel10_hell0.85_kfspatial${trans}_${rot}_${split}_gtpose.pkl --output_path ${output_path}/results/replica/predictions_gaussian_obj0.7_rel10_hell0.85_kfspatial${trans}_${rot}_${split}_gtpose.txt --label_categories replica
#         ) &
#         if (( i % 9 == 0 )); then
#             wait
#         fi
#     done
# done
# wait

# time_int=(2 3 5 10)

# i=0
# for int in "${time_int[@]}"; do
#     ((i=i+1))
#     (
#         # python main.py --dataset_path ../Datasets/Replica/ --artifact_path ../weights/RT-DETR-EGTR/VG/egtr__RT-DETR__VG__last.pth/batch__6__epochs__50_25__lr__2e-07_2e-06_2e-05__finetune/version_0/ --output_path ${output_path} --split ${split} --kf_strategy periodic --kf_interval ${int} --label_categories replica

#         python evaluate.py --dataset_path ../Datasets/Replica/ --split ${split} --prediction_path ${output_path}/replica/predictions_gaussian_obj0.7_rel10_hell0.85_kfperiodic${int}_${split}_gtpose.pkl --output_path ${output_path}/results/replica/predictions_gaussian_obj0.7_rel10_hell0.85_kfperiodic${int}_${split}_gtpose.txt --label_categories replica
#     ) &
#     if (( i % 9 == 0 )); then
#         wait
#     fi
# done
# wait

# slam trajectory
# python main.py --dataset_path ../Datasets/Replica/ --artifact_path ../weights/RT-DETR-EGTR/VG/egtr__RT-DETR__VG__last.pth/batch__6__epochs__50_25__lr__2e-07_2e-06_2e-05__finetune/version_0/ --output_path ${output_path} --split ${split} --kf_strategy none --label_categories replica &
# python main.py --dataset_path ../Datasets/Replica/ --artifact_path ../weights/RT-DETR-EGTR/VG/egtr__RT-DETR__VG__last.pth/batch__6__epochs__50_25__lr__2e-07_2e-06_2e-05__finetune/version_0/ --output_path ${output_path} --split ${split} --kf_strategy none --label_categories replica --not_use_gt_pose

# python evaluate.py --dataset_path ../Datasets/Replica/ --split ${split} --prediction_path ${output_path}/replica/predictions_gaussian_obj0.7_rel10_hell0.85_kfnone_${split}_gtpose.pkl --output_path ${output_path}/results/replica/predictions_gaussian_obj0.7_rel10_hell0.85_kfnone_${split}_gtpose.txt --label_categories replica
python evaluate.py --dataset_path ../Datasets/Replica/ --split ${split} --prediction_path ${output_path}/replica/predictions_gaussian_obj0.7_rel10_hell0.85_kfnone_${split}.pkl --output_path ${output_path}/results/replica/predictions_gaussian_obj0.7_rel10_hell0.85_kfnone_${split}.txt --label_categories replica