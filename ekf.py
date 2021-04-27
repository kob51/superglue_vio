import numpy as np
from scipy.spatial.transform import Rotation as R
import sys

sys.path.append("./data")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

from data.rotations import Quaternion

import matplotlib

matplotlib.use("TkAgg")


def skew_symmetric(v):
    """ Skew symmetric operator for a 3x1 vector. """
    v = v.flatten()
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class EKF:
    def __init__(self, gt_p, gt_v, gt_quat, scipy=True, debug=True):
        self.scipy = scipy
        self.debug = debug

        self.use_new_data = True

        # [x,y,z,vx,vy,vz,qw,qx,qy,qz]
        self.state = np.zeros(10)
        self.state[:3] = gt_p
        self.state[3:6] = gt_v
        self.state[6:] = np.roll(gt_quat, 1)

        self.state_list = np.zeros((0, 10))
        self.addToStateList()

        self.prev_time = 0

        self.prev_accel = np.zeros(3)
        self.prev_omega = np.zeros(3)

        self.g = np.array([0, 0, -9.81])

        # covariance
        self.P = np.eye(9)

        # variances
        self.sigma_a = 0.01
        self.sigma_omega = 0.01
        self.sigma_vo = 0.005
        self.sigma_lidar = 35
        self.sigma_gnss = 0.1

        # motion model noise jacobian
        self.L = np.zeros([9, 6])
        self.L[3:, :] = np.eye(6)

        # measurement model jacobian
        self.H = np.zeros([3, 9])
        self.H[:, :3] = np.eye(3)

    def setSigmaAccel(self, sig):
        self.sigma_a = sig

    def setSigmaGyro(self, sig):
        self.sigma_omega = sig

    def addToStateList(self):
        self.state_list = np.concatenate((self.state_list, self.state.reshape(1, -1)))
        if self.debug:
            print("position", self.state[:3])
            print("velocity", self.state[3:6])
            print("quat", self.state[6:])
            print()

    def IMUPrediction(self, accel, omega, dt):

        # STATE
        x_prev = self.state[:3]
        v_prev = self.state[3:6]
        q_prev = self.state[6:]  # stored as w,x,y,z

        # IMU
        a_prev = accel
        # print(a_prev, "A_prev", R_mat @ a_prev)
        omega_prev = omega

        if self.scipy:
            rotation = R.from_quat(np.roll(q_prev, -1))  # takes in x,y,z,w
            R_mat = rotation.as_matrix()
        else:
            R_mat = Quaternion(*q_prev).to_mat()
            
        print(a_prev, "A_prev", R_mat @ a_prev)
        print("Effective Acceleration:",  R_mat @ a_prev - self.g)
        
        # current x estimate
        self.state[:3] = (
            x_prev + (dt * v_prev) + 0.5 * dt ** 2 * (R_mat @ a_prev - self.g)
        )
        self.state[3:6] = v_prev + dt * (R_mat @ a_prev - self.g)

        # get current estimated rotation from imu, rotate current frame by this new value
        # (relative rotation --> right multiply)
        if self.scipy:
            self.state[6:] = np.roll(
                (rotation * R.from_euler("xyz", dt * omega_prev)).as_quat(), 1
            )
        else:
            self.state[6:] = Quaternion(euler=dt * omega_prev).quat_mult(self.state[6:])

        # error
        F = np.eye(9)
        F[0:3, 3:6] = dt * np.eye(3)
        if self.scipy:
            F[3:6, 6:9] = R_mat @ -skew_symmetric(a_prev) * dt
        else:
            F[3:6, 6:9] = R_mat.dot(-skew_symmetric(a_prev)) * dt

        # uncertainty
        Q = np.eye(6)
        Q[0:3, 0:3] = self.sigma_a * Q[0:3, 0:3]
        Q[3:6, 3:6] = self.sigma_omega * Q[3:6, 3:6]

        Q = dt ** 2 * Q

        self.P = F @ self.P @ F.T + self.L @ Q @ self.L.T

    def xyzUpdate(self, input_meas, meas_type):

        # takes input measurement of current x,y,z position

        Rot = np.eye(3)
        if meas_type == "lidar":
            Rot *= self.sigma_lidar
        if meas_type == "gnss":
            Rot *= self.sigma_gnss
        if meas_type == "vo":
            Rot *= self.sigma_vo

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + Rot)

        if not self.use_new_data:
            input_meas = self.state[:3].copy()

        delta_x = K @ (input_meas - self.state[:3])
        print(delta_x)

        self.state[:3] = self.state[:3] + delta_x[:3]
        self.state[3:6] = self.state[3:6] + delta_x[3:6]
        print(self.state, ": STATE")
        if self.scipy:
            self.state[6:] = np.roll(
                (
                    R.from_quat(np.roll(self.state[6:], -1))
                    * R.from_rotvec(delta_x[6:])
                ).as_quat(),
                1,
            )
        else:
            self.state[6:] = Quaternion(axis_angle=delta_x[6:]).quat_mult(
                self.state[6:]
            )

        self.P = (np.eye(9) - K @ self.H) @ self.P  # TODO

    def SuperGlueUpdate(self, xyz):
        self.xyzUpdate(xyz, "vo")

    def getTrajectory(self):
        # return xyz coordinates for plotting
        return self.state_list[:, :3]


# MAIN LOOP #####################
if __name__ == "__main__":
    with open("data/p1_data.pkl", "rb") as file:
        data = pickle.load(file)

    gt = data["gt"]
    imu_f = data["imu_f"]
    imu_w = data["imu_w"]
    gnss = data["gnss"]
    lidar = data["lidar"]

    C_li = np.array(
        [
            [0.99376, -0.09722, 0.05466],
            [0.09971, 0.99401, -0.04475],
            [-0.04998, 0.04992, 0.9975],
        ]
    )

    t_li_i = np.array([0.5, 0.1, 0.5])

    lidar.data = (C_li @ lidar.data.T).T + t_li_i

    # initialize with the gt position
    # don't use gt anymore for the rest of the run
    ekf = EKF(gt.p[0], gt.v[0], R.from_euler("xyz", gt.r[0]).as_quat(), debug=True)
    ekf.setSigmaAccel(0.01)
    ekf.setSigmaGyro(0.01)

    x_list = []
    x_list.append(gt.p[0])

    for k in range(1, imu_f.data.shape[0]):
        dt = imu_f.t[k] - imu_f.t[k - 1]

        accelerometer = imu_f.data[k - 1]
        gyroscope = imu_w.data[k - 1]

        # Prediction step using the current IMU reading
        ekf.IMUPrediction(accelerometer, gyroscope, dt)

        # Update step using gnss measurements
        for i in range(len(gnss.t)):
            if abs(gnss.t[i] - imu_f.t[k]) < 0.01:
                ekf.xyzUpdate(gnss.data[i], "gnss")

        # Update step using lidar measurements
        for i in range(len(lidar.t)):
            if abs(lidar.t[i] - imu_f.t[k]) < 0.01:
                ekf.xyzUpdate(lidar.data[i], "lidar")

        # add the current state estimate to our state list
        ekf.addToStateList()

    # EVALUATION STUFF
    pred = ekf.state_list[:, :3]
    ground_truth = gt.p

    gt_len = len(ground_truth)
    pred_len = len(pred)

    num = min(gt_len, pred_len)
    norms = np.linalg.norm(ground_truth[:num, :] - pred[:num, :], axis=1)
    print()
    print("- RESULTS ----------------")
    print()
    print("Total state estimates:", pred_len)
    print("Total gt poses:", gt_len)
    print()
    print("Avg Error:", np.average(norms))
    print("Max Error:", np.max(norms), np.argmax(norms))
    print("Min Error:", np.min(norms), np.argmin(norms))

    est_traj_fig = plt.figure()
    ax = est_traj_fig.add_subplot(111, projection="3d")

    ax.plot(
        ekf.state_list[:, 0],
        ekf.state_list[:, 1],
        ekf.state_list[:, 2],
        label="Estimated",
    )
    ax.plot(gt.p[:, 0], gt.p[:, 1], gt.p[:, 2], label="Ground Truth")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Estimated Trajectory")
    ax.legend()
    ax.set_zlim(-1, 5)
    plt.show()