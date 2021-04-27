import numpy as np
from quaternion import Quaternion 


class EKF:
    
    def __init__(self, initial_p_estimate, initial_v_estimate, initial_q_estimate):
        
        # TODO: Check this
        self.lidar_to_IMU_transform = np.eye(4)
        
        # Variances #TODO: Tune
        self.var_imu_f = 0.01
        self.var_imu_w = 0.01
        self.var_sp = 35
        
        # Constants
        self.g = np.array([0,0,-9.81])
        self.l_jac = np.zeros([9, 6])
        self.l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
        self.h_jac = np.zeros([3, 9])
        self.h_jac[:, :3] = np.eye(3)  # measurement model jacobian
        
        # Estimates
        self.p_est = [] #Old, and current estimate
        self.v_est = []
        self.q_est = []
        self.estimate_cov = []
        
        self.p_est.append(initial_p_estimate)
        self.v_est.append(initial_v_estimate)
        self.q_est.append(initial_q_estimate)
        self.estimate_cov.append(np.eye(9))
        
        
    def measurement_update(self, estimate_cov, y, p, v, q):
        
        #Kalman Gain
        R_cov = self.var_sp * np.eye(3)
        K = estimate_cov.dot(self.h_jac.T.dot(np.linalg.inv(self.h_jac.dot(estimate_cov.dot(self.h_jac.T)) + R_cov)))
        
        #Error in state
        d_x = K.dot(y - p)
        
        #Correction in state
        p = p + d_x[:3]
        v = v + d_x[3:6]
        q = Quaternion(axis_angle=d_x[6:]).quat_mult(q)
        
        # Update covariances
        estimate_cov = (np.eye(9) - K.dot(self.h_jac)).dot(estimate_cov)
        
        return p, v, q, estimate_cov        
        
        
    
    def imu_update(self, imu_data, imu_data_old):
        
        delta_t = imu_data.timestamp - imu_data_old.timestamp
        print(delta_t)
        
        # Update state using IMU
        # rotation = Quaternion(*(self.q_est[-1].to_numpy())).to_mat()
        rotation = self.q_est[-1].to_mat()

        acceleration = np.array([imu_data_old.accelerometer[0],imu_data_old.accelerometer[1],imu_data_old.accelerometer[2]])
        angular_velocity = np.array([imu_data_old.gyroscope[0],imu_data_old.gyroscope[1],imu_data_old.gyroscope[2]])
        
        next_p = self.p_est[-1] + delta_t * self.v_est[-1] + 0.5 * (delta_t**2) * (rotation.dot(acceleration + self.g))
        next_v = self.v_est[-1] + delta_t * (rotation.dot(acceleration - self.g))
        next_q = Quaternion(euler = delta_t * angular_velocity).quat_mult(self.q_est[-1], out="Quaternion")
        self.p_est.append(next_p)
        self.v_est.append(next_v)
        self.q_est.append(next_q)
        
        
        
        
        
        
        
        