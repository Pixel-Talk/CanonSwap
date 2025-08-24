# coding: utf-8

"""
functions for processing and transforming 3D facial keypoints
"""

import numpy as np
import torch
import torch.nn.functional as F

PI = np.pi


def headpose_pred_to_degree(pred):
    """
    pred: (bs, 66) or (bs, 1) or others
    """
    if pred.ndim > 1 and pred.shape[1] == 66:
        # NOTE: note that the average is modified to 97.5
        device = pred.device
        idx_tensor = [idx for idx in range(0, 66)]
        idx_tensor = torch.FloatTensor(idx_tensor).to(device)
        pred = F.softmax(pred, dim=1)
        degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 97.5

        return degree

    return pred


def get_rotation_matrix(pitch_, yaw_, roll_):
    """ the input is in degree
    """
    # transform to radian
    pitch = pitch_ / 180 * PI
    yaw = yaw_ / 180 * PI
    roll = roll_ / 180 * PI

    device = pitch.device

    if pitch.ndim == 1:
        pitch = pitch.unsqueeze(1)
    if yaw.ndim == 1:
        yaw = yaw.unsqueeze(1)
    if roll.ndim == 1:
        roll = roll.unsqueeze(1)

    # calculate the euler matrix
    bs = pitch.shape[0]
    ones = torch.ones([bs, 1]).to(device)
    zeros = torch.zeros([bs, 1]).to(device)
    x, y, z = pitch, yaw, roll

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([bs, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([bs, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([bs, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)  # transpose

def get_multi_rotation_matrix(pitch_, yaw_, roll_, num_steps=20):
    # Step 1: Get initial rotation matrix (from provided pitch, yaw, roll)
    initial_rotation = get_rotation_matrix(pitch_, yaw_, roll_)
    
    # Step 2: Create the final "identity" rotation matrix (no rotation)
    final_rotation = get_rotation_matrix(torch.zeros_like(pitch_), torch.zeros_like(yaw_), torch.zeros_like(roll_))
    
    # Step 3: Linearly interpolate between initial and final rotation (by interpolating pitch, yaw, roll)
    interpolated_matrices = []
    for i in range(num_steps):
        t = i / (num_steps - 1)  # interpolation parameter
        
        # Linearly interpolate pitch, yaw, and roll
        interpolated_pitch = (1 - t) * pitch_ + t * 0  # Final pitch is 0
        interpolated_yaw = (1 - t) * yaw_ + t * 0    # Final yaw is 0
        interpolated_roll = (1 - t) * roll_ + t * 0   # Final roll is 0
        
        # Generate the rotated matrix at this interpolated point
        interpolated_matrix = get_rotation_matrix(interpolated_pitch, interpolated_yaw, interpolated_roll)
        interpolated_matrices.append(interpolated_matrix)

    return interpolated_matrices