import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

# data = np.load("orig_data.npz")
data = np.load("imu_output.npz")

gt = data['gt']
if gt.shape[0] == 4:
    gt = gt.T
# pred = data['est']
accel = data['accel']
gyro = data['gyro']


est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')

print("gyro",gyro.shape)
print("accel",accel.shape)
print("gt",gt.shape)

plot_every = 5

for i in range(len(gyro)):
    ax.plot(gt[:i,0],gt[:i,1],gt[:i,2],color='r')
    if i % plot_every == 0:
        plt.waitforbuttonpress()
        print()
        print("---------------------------------")
        print()
    print()
    print(accel[i,:],"accel")
    print(gyro[i,:],"gyro")

    
