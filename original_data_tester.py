import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

data = np.load("orig_data.npz")

gt = data['gt']
pred = data['est']
accel = data['accel']
gyro = data['gyro']


est_traj_fig = plt.figure()
ax = est_traj_fig.add_subplot(111, projection='3d')

print("gyro",gyro.shape)
print("accel",accel.shape)

print("pred",pred.shape)
print("gt",gt.shape)

for i in range(len(gyro)):
    ax.plot(pred[:i,0],pred[:i,1],pred[:i,2],color='r')
    ax.plot(gt[:i,0],gt[:i,1],gt[:i,2],color='g')
    if i % 100 == 0:
        plt.waitforbuttonpress()
        print()
        print("---------------------------------")
        print()
    print()
    print(accel[i,:],"accel")
    print(gyro[i,:],"gyro")

    
