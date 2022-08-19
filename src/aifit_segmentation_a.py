# Created by Dmitriy Shin on 7/12/22 at 7:29 PM

















#!!!!!!!!!!!!!!!!!!!!!!!! IMPLEMENT

















import math
import json
import matplotlib.pyplot as plt
import time

import scipy.optimize as opt
import numpy as np
import cv2


def print_elapsed_time(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


'''
0 - pelvis
1 - right hip
2 - right knee
3 - right ankle
4 - left hip
5 - left knee
6 - left ankle
7 - mid back
8 - neck
9 - nose
10 - center head
11 - left shoulder
12 - left elbow
13 - left wrist
14 - right shoulder
15 - right elbow
16 - right wrist
'''

joint_idxs = {}
joint_idxs["pelvis"] = 0
joint_idxs["right hip"] = 1
joint_idxs["right knee"] = 2
joint_idxs["right ankle"] = 3
joint_idxs["left hip"] = 4
joint_idxs["left knee"] = 5
joint_idxs["left ankle"] = 6
joint_idxs["mid back"] = 7
joint_idxs["neck"] = 8
joint_idxs["nose"] = 9
joint_idxs["center head"] = 10
joint_idxs["left shoulder"] = 11
joint_idxs["left elbow"] = 12
joint_idxs["left wrist"] = 13
joint_idxs["right shoulder"] = 14
joint_idxs["right elbow"] = 15
joint_idxs["right wrist"] = 16
joint_idxs_key_list = list(joint_idxs.keys())
joint_idxs_val_list = list(joint_idxs.values())


# make it a function
def repetition_segmentation(active_joint_flags):  # returns an array of interval ticks

    with open("../data/moving_parts/world_coordinates_3d.json", "r") as f:
        json_data = json.load(f)

    frames_with_joints = json_data["0"]

    # poseX_joint_data - dictionary joints
    # each skeleton/pose is 17 joints
    # joints  - list of boolean values inidicating which joint to incluse in the calcucaltion
    # returns pose affinity as negative MPJPE - mean per joint position error between two poses
    def compute_pose_affinity(pose1_joint_data, pose2_joint_data, joints_to_process):
        error_acc = 0
        active_j_count = 0
        for j_idx, j_f in enumerate(joints_to_process):
            if j_f:
                active_j_count += 1
                error_vec = [pose1_joint_data[f"{j_idx}"][0] - pose2_joint_data[f"{j_idx}"][0],
                             pose1_joint_data[f"{j_idx}"][1] - pose2_joint_data[f"{j_idx}"][1],
                             pose1_joint_data[f"{j_idx}"][2] - pose2_joint_data[f"{j_idx}"][2]]
                dist_vec_length = math.sqrt(error_vec[0] ** 2 + error_vec[1] ** 2 + error_vec[2] ** 2)
                error_acc += dist_vec_length
        return -error_acc / active_j_count

    '''
    # test computation of pose affinities
    for f_idx, _ in enumerate(frames_with_joints):
        idx_first = f_idx
        # idx_second = len(frames_with_joints)-f_idx-1
        idx_second = len(frames_with_joints) - f_idx + 3
        if idx_second >= len(frames_with_joints): idx_second = 0
        aff = compute_pose_affinity(frames_with_joints[f"{idx_first}"], frames_with_joints[f"{idx_second}"],
                                    active_joint_flags)
        #print(f" First pose frame index: {idx_first}, Second pose frame index: {idx_second}, Affinity: {aff}")
    '''

    # computes autocorrelation for specific values of shrinkage and tau
    # shrinkage and tau are integers reflecting numbers of frames in pose sequence data
    def compute_autocorrelation(shrinkage, tau):
        N = len(frames_with_joints)
        affinity_acc = 0
        pose_affs = []
        ts = []
        for t in range(shrinkage, N - shrinkage - tau):
            idx_1 = t
            idx_2 = t + tau
            cur_aff = compute_pose_affinity(frames_with_joints[f"{idx_1}"], frames_with_joints[f"{idx_2}"],
                                            active_joint_flags)
            pose_affs.append(cur_aff)
            ts.append(t)
            affinity_acc += cur_aff
        Rpp = (affinity_acc / (N - 2 * shrinkage - tau))
        # plt.plot(ts, pose_affs, linestyle='--', marker='o', color='b');
        # plt.xlabel('Start Frame');
        # plt.ylabel('Pose Affinity');
        # plt.title(f'Pose Affinity vs Start Frame at Shrinkage {shrinkage}, Tau {tau}');
        # plt.show()
        return Rpp

    # computation of tau_star - initial estimate of tau assuming fixed period (repetition) of pose signal
    # iterate over shrinkage then tau
    start_time = time.time()
    max_Rpps = []
    Rpps = []
    min_tau = 30  # at 30fps value 30 equals to one second. it can be estimated from expert/template exercise
    cur_max_Rpp = (-100, None, None)
    for cur_shrinkage in range(0, int(len(frames_with_joints) / 6)):
        cur_Rpps = []
        for cur_tau in range(min_tau, int(len(frames_with_joints) / 2)):
            cur_Rpp = compute_autocorrelation(cur_shrinkage, cur_tau)
            Rpps.append((cur_Rpp, cur_tau, cur_shrinkage))
            cur_Rpps.append((cur_Rpp, cur_tau, cur_shrinkage))
            # monitor for smallest tau corresponding to maximum Rpp
            # if abs(cur_Rpp-cur_max_Rpp[0]) < sigma:
            if cur_Rpp > cur_max_Rpp[0]:
                cur_max_Rpp = (cur_Rpp, cur_tau, cur_shrinkage)
                max_Rpps.append(cur_max_Rpp)
        '''
        xs = []
        for x in cur_Rpps:
           xs.append(x[1])
           ys = []
        for y in cur_Rpps:
           ys.append(y[0])
        plt.plot(xs, ys, linestyle='--', marker='o', color='b');
        plt.xlabel('Tau');
        plt.ylabel('Autocorrelation of Pose Affinities');
        plt.title(f'Autocorrelation of Pose Affinities vs Tau at Shrinkage {cur_shrinkage}');
        plt.show()
        '''

    print(f"Reaching breakpoint sigma=0.001 took: {print_elapsed_time(start_time, time.time())}")
    # select the first period tau which is the smallest for which Rpp is max in max_Rpps
    sigma = 0.001  # defines a region of Rpp values from top value within which all Rpp are considered "equal"

    # sort by tau in descending order
    def sortValue(e):
        return e[0]

    max_Rpps.sort(key=sortValue, reverse=True)

    # save list for analysis
    with open("../max_Rpps.txt", 'w') as file_max_Rpps:
        for max_Rpp in max_Rpps:
            file_max_Rpps.write(f"{max_Rpp}\n")

    # loop from top and select max_Rpps that are within range of sigma value from the top(first)
    top_max_Rpp_value = max_Rpps[0][0]
    selected_max_Rpps = []
    for max_Rpp in max_Rpps:
        if abs(top_max_Rpp_value - max_Rpp[0]) <= sigma:
            selected_max_Rpps.append(max_Rpp)

    # save list for analysis
    with open("../sel_max_Rpps.txt", 'w') as file_sel_max_Rpps:
        for sel_max_Rpp in selected_max_Rpps:
            file_sel_max_Rpps.write(f"{sel_max_Rpp}\n")

    # from the selection select max_Rpp with the lowest tau
    def minValue(e):
        return e[1]

    Rpp = min(selected_max_Rpps, key=minValue)
    tau_star = Rpp[1]
    shrinkage_star = Rpp[2]

    print(f"Initial estimation of fixed period Tau took: {print_elapsed_time(start_time, time.time())} "
          f"for {len(frames_with_joints)}-frame video and {sum(active_joint_flags)} joints.")
    print(f"Initial fixed Tau* estimate: {tau_star} at shrinkage s*: {shrinkage_star}.")

    k_min = 5

    # computation of the t_start

    def compute_Aff_seq(t_start, i, j):
        Aff_seq = 1 / tau_star
        Aff_seq_acc = 0
        for l in range(1, tau_star + 1):  # go over frames in fixed interval tau_star
            tau_shift_i = t_start + tau_star * (i - 1)
            tau_shift_j = t_start + tau_star * (j - 1)
            Aff_seq_acc += compute_pose_affinity(frames_with_joints[f"{tau_shift_i + l}"],
                                                 frames_with_joints[f"{tau_shift_j + l}"], active_joint_flags)
        Aff_seq *= Aff_seq_acc
        return Aff_seq

    def compute_Aff_avg(t_start):
        Aff_avg = 1 / (k_min ** 2)
        Aff_seq_acc = 0
        # i and j are repetition number, not index
        for i in range(1, k_min + 1):
            for j in range(1, k_min + 1):
                Aff_seq_acc += compute_Aff_seq(t_start, i, j)
        Aff_avg *= Aff_seq_acc
        return Aff_avg

    # compute Aff_avgs for frames from 0 to tau_star/4
    Aff_avgs = []
    for f in range(int(tau_star / 2)):
        cur_Aff_avg = compute_Aff_avg(f)
        Aff_avgs.append((cur_Aff_avg, f))

    Aff_avgs.sort(key=sortValue, reverse=True)

    # save list for analysis
    with open("../Aff_avgs.txt", 'w') as file_Aff_avgs:
        for Aff_avg in Aff_avgs:
            file_Aff_avgs.write(f"{Aff_avg[0]}, {Aff_avg[1]}\n")

    # select smallest t_start for which Aff_avg has local maximum

    # loop from top and select Aff_avgs that are within range of sigma value from the top(first)
    top_Aff_avg_value = Aff_avgs[0][0]
    # print(f"Top Aff_avg: {top_Aff_avg_value}")
    selected_Aff_avgs = []
    for Aff_avg in Aff_avgs:
        if abs(top_Aff_avg_value - Aff_avg[0]) <= sigma:
            selected_Aff_avgs.append(Aff_avg)
            # print(f"Adding Aff_avg: {Aff_avg[0]}, {Aff_avg[1]}")

    # save list for analysis
    with open("../sel_Aff_avgs.txt", 'w') as file_sel_Aff_avgs:
        for sel_Aff_avg in selected_Aff_avgs:
            file_sel_Aff_avgs.write(f"{sel_Aff_avg[0]}, {sel_Aff_avg[1]}\n")

    t_start_star = min(selected_Aff_avgs, key=minValue)[1]
    print(f"Beginning or the first repetition t_start_star: {t_start_star}")

    # Drop fixed tau assumption and compute start and duration of each repetition

    def u_sample(start_frame, end_frame, num_frames_to_sample):
        # divide the interval from start to end frames into num_sample_frames
        if num_frames_to_sample <= 0 or (end_frame < start_frame):
            ggg = 0
            return (False, None)
        if (abs(end_frame - start_frame) < num_frames_to_sample):
            # print(f"start frame: {start_frame}, end frame: {end_frame}, number of frames to sample: {num_frames_to_sample}")
            # return (False, None)
            ffr = 0
        step = abs(end_frame - start_frame) / (num_frames_to_sample - 1)
        returnValue = []
        current_frame = start_frame
        returnValue.append(current_frame)
        for i in range(1, num_frames_to_sample - 2 + 1):
            current_frame += step
            returnValue.append(current_frame)
        returnValue.append(end_frame)
        return returnValue

    # dddd = u_sample(15, 19,4)
    # print(dddd)

    # modified Aff_seq_hat and Aff_avg_hat

    def compute_Aff_seq_hat(num_frames_to_sample, i_start_frame, i_end_frame, j_start_frame, j_end_frame,
                            joints_to_process):
        Aff_seq_hat = 1 / num_frames_to_sample
        Aff_seq_hat_acc = 0
        # generate sample frames for both intervals
        first_interval_frames = u_sample(i_start_frame, i_end_frame, num_frames_to_sample)
        second_interval_frames = u_sample(j_start_frame, j_end_frame, num_frames_to_sample)
        # two loops go to over all combinations of frames between two intervals
        for i_idx, i_frame in enumerate(first_interval_frames):
            # first and last will be whole numbers, i.e. integers, so skip the next operation
            i_pose = None
            if (i_idx != 0) and (i_idx != (len(first_interval_frames) - 1)):
                # check if the frame number is not discrete (whole number)
                if i_frame.is_integer():
                    # set the i pose
                    i_pose = frames_with_joints[f"{int(i_frame)}"]
                else:
                    # compute interpolated i pose
                    i_frame_floor = int(math.floor(i_frame))
                    i_frame_ceiling = int(math.ceil(i_frame))
                    i_pose_floor = frames_with_joints[f"{i_frame_floor}"]
                    i_pose_ceiling = frames_with_joints[f"{i_frame_ceiling}"]
                    floor_coeff = 1 - (i_frame - i_frame_floor)
                    ceiling_coeff = i_frame - i_frame_floor
                    # go over joints, and then their coordinates and scale them
                    first_interpolation_term = {}
                    second_interpolation_term = {}
                    scaled_joint_data = []
                    for joint_idx in range(len(i_pose_floor)):
                        first_interpolation_term[f"{joint_idx}"] = [floor_coeff * i_pose_floor[f"{joint_idx}"][0],
                                                                    floor_coeff * i_pose_floor[f"{joint_idx}"][1],
                                                                    floor_coeff * i_pose_floor[f"{joint_idx}"][2]]
                    for joint_idx in range(len(i_pose_ceiling)):
                        second_interpolation_term[f"{joint_idx}"] = [ceiling_coeff * i_pose_ceiling[f"{joint_idx}"][0],
                                                                     ceiling_coeff * i_pose_ceiling[f"{joint_idx}"][1],
                                                                     ceiling_coeff * i_pose_ceiling[f"{joint_idx}"][2]]
                    # sum first and second interpolation terms
                    i_pose = {}
                    for joint_idx in range(len(i_pose_floor)):
                        i_pose[f"{joint_idx}"] = [
                            first_interpolation_term[f"{joint_idx}"][0] + second_interpolation_term[f"{joint_idx}"][0],
                            first_interpolation_term[f"{joint_idx}"][1] + second_interpolation_term[f"{joint_idx}"][1],
                            first_interpolation_term[f"{joint_idx}"][2] + second_interpolation_term[f"{joint_idx}"][2]]
            else:
                i_pose = frames_with_joints[f"{int(i_frame)}"]
            for j_idx, j_frame in enumerate(second_interval_frames):
                # first and last will be whole numbers, i.e. integers, so skip the next operation
                j_pose = None
                if (j_idx != 0) and (j_idx != len(second_interval_frames) - 1):
                    # check if the frame number is not discrete (whole number)
                    if j_frame.is_integer():
                        j_pose = frames_with_joints[f"{int(j_frame)}"]
                    else:
                        # compute interpolated j pose
                        j_frame_floor = int(math.floor(j_frame))
                        j_frame_ceiling = int(math.ceil(j_frame))
                        j_pose_floor = frames_with_joints[f"{j_frame_floor}"]
                        j_pose_ceiling = frames_with_joints[f"{j_frame_ceiling}"]
                        floor_coeff = 1 - (j_frame - j_frame_floor)
                        ceiling_coeff = j_frame - j_frame_floor
                        # go over joints, and then their coordinates and scale them
                        first_interpolation_term = {}
                        second_interpolation_term = {}
                        scaled_joint_data = []
                        for joint_idx in range(len(j_pose_floor)):
                            first_interpolation_term[f"{joint_idx}"] = [floor_coeff * j_pose_floor[f"{joint_idx}"][0],
                                                                        floor_coeff * j_pose_floor[f"{joint_idx}"][1],
                                                                        floor_coeff * j_pose_floor[f"{joint_idx}"][2]]
                        for joint_idx in range(len(j_pose_ceiling)):
                            second_interpolation_term[f"{joint_idx}"] = [
                                ceiling_coeff * j_pose_ceiling[f"{joint_idx}"][0],
                                ceiling_coeff * j_pose_ceiling[f"{joint_idx}"][1],
                                ceiling_coeff * j_pose_ceiling[f"{joint_idx}"][2]]
                        # sum first and second interpolation terms
                        j_pose = {}
                        for joint_idx in range(len(j_pose_floor)):
                            j_pose[f"{joint_idx}"] = [
                                first_interpolation_term[f"{joint_idx}"][0] + second_interpolation_term[f"{joint_idx}"][
                                    0],
                                first_interpolation_term[f"{joint_idx}"][1] + second_interpolation_term[f"{joint_idx}"][
                                    1],
                                first_interpolation_term[f"{joint_idx}"][2] + second_interpolation_term[f"{joint_idx}"][
                                    2]]
                else:
                    j_pose = frames_with_joints[f"{int(j_frame)}"]
                Aff_seq_hat_acc += compute_pose_affinity(i_pose, j_pose, joints_to_process)

        Aff_seq_hat *= Aff_seq_hat_acc
        # print(f"{Aff_seq_hat}")
        return Aff_seq_hat

    print("Computation of repetition segmentation with non-fixed period Tau started...")
    start_time = time.time()
    num_repetitions = int(
        len(frames_with_joints) / tau_star) - 1  # try to estimate this as an extra term in optimization

    # compute number of frames based on tau_star
    num_frames_to_sample = int(tau_star / 4)
    joints_to_process_hat = [True for i in range(17)]

    # t_i_s is an array of start frame indices for repetitions (interval)
    # for each repetition end frame index is the start frame index of the following repetition (interval)
    # the size of t_i_s array should be num_repetitions + 1
    def compute_Aff_avg_hat(t_i_s: []):  # objective function
        Aff_avg = 1 / (num_repetitions ** 2)
        Aff_seq_acc = 0
        # i and j are repetition index, not number
        for i in range(0, num_repetitions):
            i_start_frame = t_i_s[i]
            i_end_frame = t_i_s[i + 1]
            for j in range(0, num_repetitions):
                j_start_frame = t_i_s[j]
                j_end_frame = t_i_s[j + 1]
                Aff_seq_acc += compute_Aff_seq_hat(num_frames_to_sample, i_start_frame, i_end_frame, j_start_frame,
                                                   j_end_frame, joints_to_process_hat)
        Aff_avg *= Aff_seq_acc
        print(f"{t_i_s}")
        print(f"{Aff_avg}")
        return -Aff_avg

    # use t_start_star and tau_star for initialization of non-linear constrained optimization
    t_i_s_init = []
    for i in range(num_repetitions + 1):
        t_i_s_init.append(float(t_start_star + i * tau_star))

    # t_i_s_init[1] = 50
    constraints = []

    def global_frames_interval_constraint(x):  # length of array x is num_repetitions+1, output array is num_repetitions
        return x

    constraint = opt.NonlinearConstraint(global_frames_interval_constraint, lb=0, ub=len(frames_with_joints))

    constraints.append(constraint)

    # constraint to ensure t_i+1 - t_i > delta
    # it effectively ensures that start_frame < end_frame
    # delta = num_frames_to_sample  # # of frames between end frame and start frame of the next repetition
    # delta = tau_star -10  # # of frames between end frame and start frame of the next repetition
    delta = 10

    def consecutive_frames_constraint(x):  # length of array x is num_repetitions+1, output array is num_repetitions
        return_value = []
        for i in range(len(x) - 1):
            return_value.append(x[i + 1] - x[i])
        return return_value

    #    cc_constraint = opt.NonlinearConstraint(consecutive_frames_constraint, lb=delta, ub=tau_star + delta)
    cc_lower_bound = np.full((num_repetitions), delta)
    cc_upper_bound = np.full((num_repetitions), tau_star + delta)
    # cc_lower_bound = np.full((num_repetitions), 1)
    # cc_upper_bound = np.full((num_repetitions), delta)
    # cc_constraint = opt.NonlinearConstraint(consecutive_frames_constraint, lb=0.0, ub=delta)
    cc_constraint = opt.NonlinearConstraint(consecutive_frames_constraint, lb=cc_lower_bound, ub=cc_upper_bound)
    # constraints.append({"type": "ineq", "conseq_frames": cc_constraint})
    constraints.append(cc_constraint)

    res = opt.minimize(compute_Aff_avg_hat, x0=t_i_s_init, constraints=constraints,# method='L-BFGS-B',
                       options={'eps': 30})#, 'maxiter': 20})  # , method='TNC')

    print(res)
    print(res.x)

    print(
        f"Computation of repetition segmentation with non-fixed period Tau took: {print_elapsed_time(start_time, time.time())}")

    return True, frames_with_joints, res.x

    #if res.success:
    #    return True, frames_with_joints, res.x
    #else:
    #    return False, None, None


# video overlay function
def video_overlay(frames_with_joints, interval_ticks, active_joint_flags):
    joint_info = ""
    for act_flag_idx in range(len(active_joint_flags)):
        if active_joint_flags[act_flag_idx]:
            position = joint_idxs_val_list.index(act_flag_idx)
            joint_info += f" {joint_idxs_key_list[position]}"

    # video overlay

    # if res.success:
    interval_ticks = [round(interval_ticks[i]) for i in range(len(interval_ticks))]
    cap = cv2.VideoCapture("../data/moving_parts/video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'H264')  # encodes H264 even though gives errors
    out = cv2.VideoWriter(f"../out/{joint_info}_overlay_video.mp4", fourcc, 30.0, (2400, 1080))
    h_tick_length = round(2400 / len(frames_with_joints))

    # processed_ticks_flags = [False for i in range(len(interval_ticks))]
    cap.set(1, 1)  # set the first frame
    ret, frame = cap.read()
    frame_count_idx = 0
    current_active_tick_idx = 0
    # for idx_tick in range(len(interval_ticks)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 2
    while ret:
        current_active_tick = interval_ticks[current_active_tick_idx]
        # draw tick history
        current_prev_tick_idx = 0
        for prev_frame_idx in range(frame_count_idx):
            # compute x position
            x_pos = round(h_tick_length * prev_frame_idx)
            org = (x_pos, 200)
            current_prev_tick = interval_ticks[current_prev_tick_idx]
            if prev_frame_idx == current_prev_tick:
                # draw a vertical  tick
                color = (0, 0, 255)
                thickness = 5
                cv2.putText(frame, '|', org, font, fontScale, color, thickness, cv2.LINE_AA)
                if current_prev_tick_idx < (
                        len(interval_ticks) - 1):  # stop incrementing current_active_tick_idx when it reaches the end
                    current_prev_tick_idx += 1
            else:
                # draw a horizontal tick
                color = (255, 0, 0)
                thickness = 4
                cv2.putText(frame, '-', org, font, fontScale, color, thickness, cv2.LINE_AA)
        org = (round((frame_count_idx - 1) * h_tick_length), 200)
        if frame_count_idx == current_active_tick:
            # draw a vertical  tick
            color = (0, 0, 255)
            thickness = 5
            cv2.putText(frame, '|', org, font, fontScale, color, thickness, cv2.LINE_AA)
            if current_active_tick_idx < (
                    len(interval_ticks) - 1):  # stop incrementing current_active_tick_idx when it reaches the end
                current_active_tick_idx += 1
        else:
            # draw a horizontal tick
            color = (255, 0, 0)
            thickness = 4
            cv2.putText(frame, '-', org, font, fontScale, color, thickness, cv2.LINE_AA)
        # draw joint info
        org = (50, 50)
        color = (0, 255, 0)
        thickness = 2
        fontScale = 1
        cv2.putText(frame, joint_info, org, font, fontScale, color, thickness, cv2.LINE_AA)

        out.write(frame)
        frame_count_idx += 1
        ret, frame = cap.read()
    cap.release()
    out.release()


# MAIN SECTION
if __name__ == "__main__":
    active_joint_flags = [False for i in range(17)]
    # active_joint_flags = [True for i in range(17)]
    #active_joint_flags[joint_idxs["right shoulder"]] = True
    #active_joint_flags[joint_idxs["left hip"]] = True
    #active_joint_flags[joint_idxs["right wrist"]] = True
    active_joint_flags[joint_idxs["left wrist"]] = True

    success, frames_with_joints, interval_ticks = repetition_segmentation(active_joint_flags)
    if not success:
        exit(1)
    video_overlay(frames_with_joints, interval_ticks, active_joint_flags)

    d = 0

