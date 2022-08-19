# Created by Dmitriy Shin on 7/14/22 at 11:53 AM

from tslearn.metrics import dtw as tslearn_dtw
import json
import copy

# "abs_trunk_flexion_extension"
# "relative_r_trunk_flexion_extension"
# "relative_l_trunk_flexion_extension"
# "abs_trunk_lateral_bending"
# "relative_trunk_lateral_bending"
# "relative_trunk_torsion"
# "relative_midspine_flexion_extension"
# "relative_midspine_lateral_bending"
# "r_shoulder_flexion_extension"
# "l_shoulder_flexion_extension"
# "r_shoulder_abduction_adduction"
# "l_shoulder_abduction_adduction"
# "relative_r_elbow_angles"
# "relative_l_elbow_angles"
# "relative_r_knee_angles"
# "relative_l_knee_angles"





with open("../data/IMG_0031_angles.json", "r") as f:
    json_data = json.load(f)

abs_trunk_flexion_extension = json_data["abs_trunk_flexion_extension"]
relative_l_elbow_angles = json_data["relative_l_elbow_angles"]
relative_r_elbow_angles = json_data["relative_r_elbow_angles"]

seq_len = len(relative_l_elbow_angles)
seq = []
for frame_idx in range(seq_len):
    seq.append([relative_l_elbow_angles[frame_idx], relative_r_elbow_angles[frame_idx]])

seq2 = [3, 4, 8, 9, 22, 77, 54, 98, 76, 54, 23, 3, 4, 8, 9,54, 23, 3, 4, 7, 9]
template3 = [3, 4, 8, 9]
seq2_len = len(seq2)

template1 = copy.deepcopy(abs_trunk_flexion_extension[0:60])
template2 = copy.deepcopy(seq[0:60])

results = []

min_window_size = 3
max_window_size = 6



for seq_start_frame_idx in range(seq2_len):
    for window_size in range(min_window_size, max_window_size):
        if (seq_start_frame_idx + window_size) > seq2_len:
            break;
        seq_window = seq2[seq_start_frame_idx:(seq_start_frame_idx+window_size)]
        dist = tslearn_dtw(template3, seq_window)
        results.append((seq_start_frame_idx, window_size, dist))


def sortValue(e):
    return e[2]

results.sort(key=sortValue, reverse=False)


#dist = tslearn_dtw(template1, abs_trunk_flexion_extension[2:552])
#print(dist)


dist = tslearn_dtw(template2, seq[0:60])
bb=0




# set action template

# set sequence for analysis


# for each start frame in the sequence
# set smallest and largest window size
# iterate from smallest to largest window and store start_frame, window size, and distance values









