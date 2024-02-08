import numpy as np
import matplotlib.pyplot as plt
from util import eular_to_SO3

seq = '05'
gt_data = np.loadtxt('/home/smeet/catkin_ws/src/Visual-Inertial-Odometry/Dataset/poses/local_poses/' + f'{seq}.txt')
output_data = np.loadtxt('/home/smeet/catkin_ws/src/Visual-Inertial-Odometry/DeepVO/output/output_6dof_' + f'{seq}.txt')
figs_path = '/home/smeet/catkin_ws/src/Visual-Inertial-Odometry/DeepVO/figs/' + seq + '/'

H_new = np.identity(4, dtype=np.float32)
H_new2 = np.identity(4, dtype=np.float32)

coordinate_transform = np.array([[0, 0, 1, 0],
                                 [-1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, 0, 1]], dtype=np.float32) # x axis: -90.0, y axis: 0.0, z axis: -90.0 

gt_list = []
output_list = []

x_true, x_output = [], []
y_true, y_output = [], []
z_true, z_output = [], []
roll_true, roll_output = [], []
pitch_true, pitch_output = [], []
yaw_true, yaw_output = [], []

for i in range(gt_data.shape[0]):
    H_rel = np.identity(4)
    
    poses = gt_data[i, :]
    rotation = poses[:3]
    rot_mat = eular_to_SO3(rotation)
    translation = poses[3:]
    
    H_rel[:3, :3] = rot_mat
    H_rel[:3, 3] = translation

    H_new = np.dot(H_new, H_rel)
    body_pose = np.dot(coordinate_transform, H_new)

    position = body_pose[:3, 3]
    
    x_true.append(translation[0])
    y_true.append(translation[1])
    z_true.append(translation[2])
    
    roll_true.append(rotation[0])
    pitch_true.append(rotation[1])
    yaw_true.append(rotation[2])
    
    gt_list.append(position)

gt_list = np.array(gt_list)
x_true = np.array(x_true)
y_true = np.array(y_true)
z_true = np.array(z_true)
roll_true = np.array(roll_true)
pitch_true = np.array(pitch_true)
yaw_true = np.array(yaw_true)

for i in range(output_data.shape[0]):
    H_rel = np.identity(4)
    poses = output_data[i, :]
    rotation = poses[:3]
    rot_mat = eular_to_SO3(rotation)
    translation = poses[3:]
    H_rel[:3, :3] = rot_mat
    H_rel[:3, 3] = translation

    H_new2 = np.dot(H_new2, H_rel)
    body_pose = np.dot(coordinate_transform, H_new2)

    position = body_pose[:3, 3]
    
    x_output.append(translation[0])
    y_output.append(translation[1])
    z_output.append(translation[2])
    
    roll_output.append(rotation[0])
    pitch_output.append(rotation[1])
    yaw_output.append(rotation[2])
    
    output_list.append(position)

output_list = np.array(output_list)
x_output = np.array(x_output)
y_output = np.array(y_output)
z_output = np.array(z_output)
roll_output = np.array(roll_output)
pitch_output = np.array(pitch_output)
yaw_output = np.array(yaw_output)

#==================================================
plt.figure(0)
plt.plot(gt_list[:, 0], gt_list[:, 1], 'r', label='Ground Truth')
plt.plot(output_list[:, 0], output_list[:, 1], 'b', label='Model Output')
plt.plot(gt_list[:, 0][0], gt_list[:, 1][0], 'ko', label='start')
plt.title(f'sequence: {seq}')
plt.legend()
plt.savefig(figs_path + seq)

fig1 = plt.figure(1)
fig1.subplots_adjust(hspace=0.5, wspace=0.5)
ax1 = fig1.add_subplot(311)
ax1.set_xlabel('time [sec]')
ax1.set_ylabel('distance [m]')
ax1.plot(x_true, 'r', label='x: Ground Truth')
ax1.plot(x_output, 'b', label='x: Model Output')
ax1.grid(True)
ax1.legend()

ax2 = fig1.add_subplot(312)
ax2.set_xlabel('time [sec]')
ax2.set_ylabel('distance [m]')
ax2.plot(y_true, 'r', label='y: Ground Truth')
ax2.plot(y_output, 'b', label='y: Model Output')
ax2.grid(True)
ax2.legend()

ax3 = fig1.add_subplot(313)
ax3.set_xlabel('time [sec]')
ax3.set_ylabel('distance [m]')
ax3.plot(z_true, 'r', label='z: Ground Truth')
ax3.plot(z_output, 'b', label='z: Model Output')
ax3.grid(True)
ax3.legend()
fig1.savefig(figs_path + seq + '_translation.png')

#==================================================
fig2 = plt.figure(2)
fig2.subplots_adjust(hspace=0.5, wspace=0.5)
ax4 = fig2.add_subplot(311)
ax4.set_xlabel('time [sec]')
ax4.set_ylabel('angle [rad]')
ax4.plot(roll_true, 'r', label='roll: Ground Truth')
ax4.plot(roll_output, 'b', label='roll: Model Output')
ax4.grid(True)
ax4.legend()

ax5 = fig2.add_subplot(312)
ax5.set_xlabel('time [sec]')
ax5.set_ylabel('angle [rad]')
ax5.plot(pitch_true, 'r', label='pitch: Ground Truth')
ax5.plot(pitch_output, 'b', label='pitch: Model Output')
ax5.grid(True)
ax5.legend()

ax6 = fig2.add_subplot(313)
ax6.set_xlabel('time [sec]')
ax6.set_ylabel('angle [rad]')
ax6.plot(yaw_true, 'r', label='yaw: Ground Truth')
ax6.plot(yaw_output, 'b', label='yaw: Model Output')
ax6.grid(True)
ax6.legend()
fig2.savefig(figs_path + seq + '_rotation.png')

print('Done')