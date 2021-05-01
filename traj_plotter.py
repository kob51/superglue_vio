import numpy as np

from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d

import matplotlib
matplotlib.use("TkAgg")

import glob
import os

from scipy.spatial.transform import Rotation as R


if __name__ == "__main__":
    ##################################
    path = 'output/'
    filename = None
    ##################################

    if filename is None:
        list_of_files = glob.glob(path+'*') # * means all if need specific format then *.csv
        npz_path = max(list_of_files, key=os.path.getctime)
    else:
        npz_path = path + filename

    print()
    print('Data file:', npz_path)
    print()
    trajectories = np.load(npz_path)

    # 3 x N arrays of trajectories
    gt_3d = trajectories['gt'][:3]
    vo_3d = trajectories['vo'][:3]
    vio_3d = trajectories['vio'][:3]

    gt_2d = trajectories['gt'][:2]
    vo_2d = trajectories['vo'][:2]
    vio_2d = trajectories['vio'][:2]

    print(gt_3d[:,:5].T,vo_3d[:,:5].T)

    # Initialize plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")


    ax.plot3D(
        gt_3d[0],
        gt_3d[1],
        gt_3d[2],
        "red",
        label="Ground Truth",
    )

    ax.plot3D(
        vo_3d[0],
        vo_3d[1],
        vo_3d[2],
        "green",
        label="Estimated (VO)",
    )

    ax.plot3D(
        vio_3d[0],
        vio_3d[1],
        vio_3d[2],
        "blue",
        label="Estimated (VIO)",
    )

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.legend()

    vo_errors_3d = np.linalg.norm(gt_3d - vo_3d,axis=0)
    vio_errors_3d = np.linalg.norm(gt_3d - vio_3d,axis=0)


    print("- 3D RESULTS ---------------")
    print()
    print("VO")
    print("Avg Error:",np.mean(vo_errors_3d))
    print("Min Error:",np.min(vo_errors_3d),np.argmin(vo_errors_3d))
    print("Max Error:",np.max(vo_errors_3d),np.argmax(vo_errors_3d))
    print()
    print("VIO")
    print("Avg Error:",np.mean(vio_errors_3d))
    print("Min Error:",np.min(vio_errors_3d),np.argmin(vio_errors_3d))
    print("Max Error:",np.max(vio_errors_3d),np.argmax(vio_errors_3d))

    plt.show()

    

    vo_errors_2d = np.linalg.norm(gt_2d - vo_2d,axis=0)
    vio_errors_2d = np.linalg.norm(gt_2d - vio_2d,axis=0)

    print()
    print("- 2D RESULTS ---------------")
    print()
    print("VO")
    print("Avg Error:",np.mean(vo_errors_2d))
    print("Min Error:",np.min(vo_errors_2d),np.argmin(vo_errors_2d))
    print("Max Error:",np.max(vo_errors_2d),np.argmax(vo_errors_2d))
    print()
    print("VIO")
    print("Avg Error:",np.mean(vio_errors_2d))
    print("Min Error:",np.min(vio_errors_2d),np.argmin(vio_errors_2d))
    print("Max Error:",np.max(vio_errors_2d),np.argmax(vio_errors_2d))

    plt.plot(gt_2d[0],gt_2d[1],color='r',label="Ground Truth")
    plt.plot(vo_2d[0],vo_2d[1],color='g',label="Estimated (VO)")
    plt.plot(vio_2d[0],vio_2d[1],color='b',label="Estimated (VIO)")
    plt.legend()
    plt.show()


    