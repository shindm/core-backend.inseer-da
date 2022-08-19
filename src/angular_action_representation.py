import math
import json
import matplotlib.pyplot as plt
import time
import numpy as np
import matplotlib.cm as cm
from aifit_segmentation_j import repetition_segmentation
from aifit_segmentation_j import video_overlay
from aifit_segmentation_j import joint_idxs
from aifit_segmentation_j import joint_idxs_key_list
from aifit_segmentation_j import joint_idxs_val_list



def print_elapsed_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


active_joint_angle_flags = [False for i in range(8)]
# active_joint_angle_flags = [True for i in range(8)]
joint_angle_idxs = {}
joint_angle_idxs["Trunk_Flexion_Angle"] = 0
joint_angle_idxs["Neck_Flexion_Angle"] = 1
joint_angle_idxs["L_Shoulder_Elevation_Angle"] = 2
joint_angle_idxs["R_Shoulder_Elevation_Angle"] = 3
joint_angle_idxs["L_Elbow_Angle"] = 4
joint_angle_idxs["R_Elbow_Angle"] = 5
joint_angle_idxs["R_Knee_Angle"] = 6
joint_angle_idxs["L_Knee_Angle"] = 7
joint_angle_idxs_key_list = list(joint_angle_idxs.keys())
joint_angle_idxs_val_list = list(joint_angle_idxs.values())


#active_joint_angle_flags[joint_idxs["L_Elbow_Angle"]] = True
active_joint_angle_flags[joint_angle_idxs["R_Elbow_Angle"]] = True
# active_joint_flags[joint_idxs["L_Shoulder_Elevation_Angle"]] = True

with open("../data/moving_parts/angle_values.json", "r") as f:
    json_data = json.load(f)

# convert from list of joint angles with frames to frames with joint angles
frames_with_joint_angles = {}
for idx_frame in range(len(json_data["Trunk_Flexion_Angle"])):
    frame_data = {}
    for idx_joint_angle, joint_angle in enumerate(json_data):
        frame_data[idx_joint_angle] = json_data[joint_angle][idx_frame]
    frames_with_joint_angles[idx_frame] = frame_data

ddd = 0

# compute segmentaion based on coordinate pose affinities
#interval_ticks = [30.09441333, 90.88494459, 182.87904406, 271.7079706, 336.99939863, 404.43851159, 472.40040174,
#                  544.93499516]

active_joint_flags = [False for i in range(17)]
# active_joint_flags = [True for i in range(17)]
# active_joint_flags[joint_idxs["left wrist"]] = True
active_joint_flags[joint_idxs["right wrist"]] = True

success, frames_with_joints, interval_ticks = repetition_segmentation(active_joint_flags)
if not success:
    exit(1)
video_overlay(frames_with_joints, interval_ticks, active_joint_flags)

# form intervals
intervals = []
# start_frame = 0
for tick_idx in range(len(interval_ticks) - 1):
    intervals.append((interval_ticks[tick_idx], interval_ticks[tick_idx + 1]))



# plot angle waveforms for each interval
fig, stacking = plt.subplots(figsize=(10, 8))
colors = iter(cm.rainbow(np.linspace(0, 1, len(intervals))))
for interval in intervals:
    start_frame = interval[0]
    end_frame = interval[1]
    xs = []
    ys = []
    idx_x = 0
    for idx_frame in range(round(start_frame), round(end_frame)):
        y = frames_with_joint_angles[idx_frame][joint_angle_idxs["R_Elbow_Angle"]]
        ys.append(y)
        xs.append(idx_x)
        idx_x += 1
    stacking.scatter(xs, ys, c=next(colors), s=40, alpha=0.9)#, cmap=plt.cm.Set1)
fig.show()


# consider computing an averaged out across all repetitions action waveform
# since intervals are of different length, a sampling is needed
# this may help to understand the action
# statistics for duration, largest/lowest angles of action repetitions cam be useful to measure consistency
# for instance we can measure how many times an extreme joint angle occurs in action to assess potential injury


# plot normalized waveforms, i.e. the angular action profile
normalized_interval_angles = []
# find an interval of smallest duration
def intervalDuration(e):
    return abs(e[1]-e[0])

min_interval = min(intervals, key=intervalDuration)
min_interval_duration = intervalDuration(min_interval)

def sample_interpolated_angles(start_frame: int, end_frame: int, num_frames_to_sample: int):
    # divide the interval from start to end frames into num_sample_frames
    if num_frames_to_sample <= 0 or (end_frame < start_frame):
        ggg = 0
        return (False, None)
    if (abs(end_frame - start_frame) < num_frames_to_sample):
        #print(f"start frame: {start_frame}, end frame: {end_frame}, number of frames to sample: {num_frames_to_sample}")
        # return (False, None)
        ffr = 0
    step = abs(end_frame - start_frame) / (num_frames_to_sample - 1)
    sampledFrames = []
    current_frame = start_frame
    sampledFrames.append(current_frame)
    for i in range(1, num_frames_to_sample - 1):
        current_frame += step
        sampledFrames.append(current_frame)
    sampledFrames.append(end_frame)
    #compute interpolated angles
    returnValue = []
    for idx, frame in enumerate(sampledFrames):
        # first and last will be whole numbers, i.e. integers, so skip the next operation
        current_frame_angles = None
        if (idx != 0) and (idx != len(sampledFrames) - 1):
            # check if the frame number is not discrete (whole number)
            if frame.is_integer():
                # set the i pose
                current_frame_angles = frames_with_joint_angles[int(frame)]
            else:
                # compute interpolated frame angles
                current_frame_floor = int(math.floor(frame))
                current_frame_ceiling = int(math.ceil(frame))
                angle_floor = frames_with_joint_angles[current_frame_floor]
                angle_ceiling = frames_with_joint_angles[current_frame_ceiling]
                floor_coeff = 1 - (frame - current_frame_floor)
                ceiling_coeff = frame - current_frame_floor
                # go over angles and scale them
                first_interpolation_term = {}
                second_interpolation_term = {}
                for joint_idx in range(len(angle_floor)):
                    first_interpolation_term[joint_idx] = floor_coeff * angle_floor[joint_idx]
                for joint_idx in range(len(angle_ceiling)):
                    second_interpolation_term[joint_idx] = ceiling_coeff * angle_ceiling[joint_idx]
                # sum first and second interpolation terms
                current_frame_angles = {}
                for joint_idx in range(len(angle_floor)):
                    current_frame_angles[joint_idx] = first_interpolation_term[joint_idx] + second_interpolation_term[joint_idx]
        else:
            current_frame_angles = frames_with_joint_angles[int(frame)]

        returnValue.append(current_frame_angles)
    return returnValue


for interval in intervals:
    interval_angles = sample_interpolated_angles(interval[0], interval[1], int(min_interval_duration))
    normalized_interval_angles.append(interval_angles)


# plot normalized angular action profile for right wrist
fig, stacking = plt.subplots(figsize=(10, 8))
colors = iter(cm.rainbow(np.linspace(0, 1, len(normalized_interval_angles))))
for joint_angle_interval in normalized_interval_angles:
    xs = []
    ys = []
    for frame_idx, frame_angles in enumerate(joint_angle_interval):
        frame_angle = frame_angles[joint_angle_idxs["R_Elbow_Angle"]]
        xs.append(frame_idx)
        ys.append(frame_angle)
    stacking.scatter(xs, ys, c=next(colors), s=40, alpha=0.9)#, cmap=plt.cm.Set1)
fig.show()










rtt=0
















