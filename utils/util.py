import numpy as np
import math
import torch

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0


# Determine whether the matrix R is a valid rotation matrix
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculate the relative rotation (Eular angles) and translation from two pose matrices
# Rt1: a list of 12 floats
# Rt2: a list of 12 floats


def cal_rel_pose(Rt1, Rt2):
    Rt1 = np.reshape(np.array(Rt1), (3, 4))
    Rt1 = np.concatenate((Rt1, np.array([[0, 0, 0, 1]])), 0)
    
    Rt2 = np.reshape(np.array(Rt2), (3, 4))
    Rt2 = np.concatenate((Rt2, np.array([[0, 0, 0, 1]])), 0)

    # Calculate the relative transformation Rt_rel
    Rt1_inv = np.linalg.inv(Rt1)
    Rt_rel = Rt1_inv @ Rt2

    R_rel = Rt_rel[:3, :3]
    t_rel = Rt_rel[:3, 3]
    assert (isRotationMatrix(R_rel))

    # Extract the Eular angle from the relative rotation matrix
    x, y, z = SO3_to_eular(R_rel)
    theta = [x, y, z]

    pose_rel_6 = np.concatenate((theta, t_rel))

    assert (pose_rel_6.shape == (6,))
    return pose_rel_6


# Calculate the 3x4 transformation matrix from Eular angles and translation vector
# pose: (3 angles, 3 translations)
def pose_to_SE3(pose):
    H = np.identity(4, dtype=np.float32)
    R = eular_to_SO3(pose[:3])
    t = pose[3:].reshape(-1)
    H[:3, :3] = R
    H[:3, 3] = t
    return H

# Calculate the rotation matrix from eular angles (roll, yaw, pitch)


def eular_to_SO3(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ], dtype=np.float32)
    
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ], dtype=np.float32)
    
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ], dtype=np.float32)
    
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


# Calculate the eular angles (roll, yaw, pitch) from a rotation matrix
def SO3_to_eular(matrix):

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[0, 0]*M[0, 0] + M[1, 0]*M[1, 0])
    ay = math.atan2(-M[2, 0], cy)  # calculate pitch angle

    if ay < -math.pi/2 + _EPS and ay > -math.pi/2 - _EPS:  # pitch = -90 deg
        ax = 0
        az = math.atan2(-M[1, 2], -M[0, 2])
    elif ay < math.pi/2 + _EPS and ay > math.pi/2 - _EPS:
        ax = 0
        az = math.atan2(M[1, 2], M[0, 2])
    else:
        ax = math.atan2(M[2, 1], M[2, 2])
        az = math.atan2(M[1, 0], M[0, 0])

    return np.array([ax, ay, az])


def torch_to_SO3(theta):
    R_x = torch.tensor([[1, 0, 0],
                        [0, torch.cos(theta[0]), -torch.sin(theta[0])],
                        [0, torch.sin(theta[0]), torch.cos(theta[0])]], dtype=torch.float32)
    
    R_y = torch.tensor([[torch.cos(theta[1]), 0, torch.sin(theta[1])],
                        [0, 1, 0],
                        [-torch.sin(theta[1]), 0, torch.cos(theta[1])]], dtype=torch.float32)
    
    R_z = torch.tensor([[torch.cos(theta[2]), -torch.sin(theta[2]), 0],
                        [torch.sin(theta[2]), torch.cos(theta[2]), 0],
                        [0, 0, 1]], dtype=torch.float32)

    R = torch.matmul(R_z, torch.matmul(R_y, R_x))
    
    return R


def torch_to_SE3(pose):
    H = torch.eye(4, dtype=torch.float32)
    R = torch_to_SO3(pose[:3])
    t = pose[3:]
    H[:3, :3] = R
    H[:3, 3] = t
    return H


def torch_to_euler_angles(R):
    # Assuming R is a 3x3 rotation matrix
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.tensor(0.0)

    return torch.tensor([x, y, z], dtype=torch.float32)
