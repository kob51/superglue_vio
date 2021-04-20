import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('./data')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def skew_symmetric(v):
    """ Skew symmetric operator for a 3x1 vector. """
    return np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]]
    )

class EKF:
    def __init__(self,gt_p,gt_v,gt_r):
        # [x,y,z,vx,vy,vz,qw,qx,qy,qz]
        self.state = np.zeros(10)
        self.state[:3] = gt_p
        self.state[3:6] = gt_v
        self.state[6:] = np.roll(gt_r,1)

        # print(self.state[6:])

        self.prev_time = 0

        self.prev_accel = np.zeros(3)
        self.prev_omega = np.zeros(3)

        self.g = np.array([0,0,-9.81])

        # covariance
        self.P = np.eye(9)

        # variances
        self.sigma_a = 0.01
        self.sigma_omega = 0.01
        self.sigma_vo = 0
        self.sigma_lidar = 35
        self.sigma_gnss = 0.1

        # motion model noise jacobian
        self.L = np.zeros([9, 6])
        self.L[3:, :] = np.eye(6)  
        
        # measurement model jacobian
        self.H = np.zeros([3, 9])
        self.H[:, :3] = np.eye(3)  

    def IMUPrediction(self,dt,accel,omega):

        # STATE
        x_prev = self.state[:3]
        v_prev = self.state[3:6]
        q_prev = self.state[6:]    # stored as w,x,y,z


        # IMU
        a_prev = accel
        omega_prev = omega

        # print("q_prev [wxyz]",q_prev)
        # print("rolled [xyzw]",np.roll(q_prev,-1))
        rotation = R.from_quat(np.roll(q_prev,-1)) # takes in x,y,z,w
        R_mat = rotation.as_matrix()

        # current x estimate
        self.state[:3] = x_prev + (dt * v_prev) + 0.5 * dt**2 * (R_mat @ a_prev + self.g)
        self.state[3:6] = v_prev + dt * (R_mat @ a_prev - self.g)
        self.state[6:] = np.roll((R.from_euler('xyz',dt * omega_prev) * rotation).as_quat(),1)

        # error
        F = np.eye(9)
        F[0:3,3:6] = dt * np.eye(3)
        F[3:6,6:9] = R_mat @ -skew_symmetric(a_prev) * dt

        # print("F",F)

        # uncertainty
        Q = np.eye(6)
        Q[0:3,0:3] = self.sigma_a * Q[0:3,0:3]
        Q[3:6,3:6] = self.sigma_omega * Q[3:6,3:6]

        Q = dt**2 * Q

        self.P = F @ self.P @ F.T + self.L @ Q @ self.L.T

    def SuperGlueUpdate(self,t_vec):
        R = self.sigma_vo * np.eye(3)

        K = self.P @ self.H @ np.linalg.inv(self.H @ self.P @ self.H.T + R)

    def lidarUpdate(self,input_meas):
        R_ = self.sigma_lidar * np.eye(3)

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R_)

        delta_x = K @ (input_meas - self.state[:3])

        self.state[:3] = self.state[:3] + delta_x[:3]
        self.state[3:6] = self.state[3:6] + delta_x[3:6]
        self.state[6:] = np.roll((R.from_rotvec(delta_x[6:]) * R.from_quat(np.roll(self.state[6:],-1))).as_quat(),1)

        self.P = (np.eye(9) - K @ self.H) @ self.P

    def gnssUpdate(self,input_meas):
        R_ = self.sigma_gnss * np.eye(3)

        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + R_)

        delta_x = K @ (input_meas - self.state[:3])

        self.state[:3] = self.state[:3] + delta_x[:3]
        self.state[3:6] = self.state[3:6] + delta_x[3:6]
        self.state[6:] = np.roll((R.from_rotvec(delta_x[6:]) * R.from_quat(np.roll(self.state[6:],-1))).as_quat(),1)

        

        self.P = (np.eye(9) - K @ self.H) @ self.P


if __name__ == "__main__":
    with open('data/p1_data.pkl', 'rb') as file:
        data = pickle.load(file)

    gt = data['gt']
    imu_f = data['imu_f']
    imu_w = data['imu_w']
    gnss = data['gnss']
    lidar = data['lidar']

    C_li = np.array([
        [ 0.99376, -0.09722,  0.05466],
        [ 0.09971,  0.99401, -0.04475],
        [-0.04998,  0.04992,  0.9975 ]
    ])

    t_li_i = np.array([0.5, 0.1, 0.5])

    lidar.data = (C_li @ lidar.data.T).T + t_li_i

    ekf = EKF(gt.p[0],gt.v[0],R.from_euler('xyz',gt.r[0]).as_quat())

    x_list = []
    x_list.append(gt.p[0])

    for k in range(1, imu_f.data.shape[0]):
        dt = imu_f.t[k] - imu_f.t[k-1] 

        ekf.IMUPrediction(dt,imu_f.data[k-1],imu_w.data[k-1])

        

        for i in range(len(gnss.t)):
            if abs(gnss.t[i] - imu_f.t[k]) < 0.01:
                ekf.gnssUpdate(gnss.data[i])

        for i in range(len(lidar.t)):
            if abs(lidar.t[i] - imu_f.t[k]) < 0.01:
                ekf.lidarUpdate(lidar.data[i])

        x_list.append(ekf.state[:3])

        print("position",ekf.state[:3])
        print("velocity",ekf.state[3:6])
        print("quat",ekf.state[6:])
        print()

        if k > 100:
            fuck
