set -e
scenes=("apartment_0" "apartment_1" "apartment_2" "office_0" "office_1" "office_2" "office_3" "office_4" "room_0" "room_1" "room_2" "hotel_0" "frl_apartment_0" "frl_apartment_1" "frl_apartment_2" "frl_apartment_3" "frl_apartment_4" "frl_apartment_5")

for scene in "${scenes[@]}"; do
    python convert_SLAM_trajectory.py --replica_path /warehouse/howardkhh/FROSS/Replica --trajectory ~/ORB_SLAM3/CameraTrajectory_${scene}.txt --scene ${scene}
done
