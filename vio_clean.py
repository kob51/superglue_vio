import carla
import numpy as np
import random
import time
import tkinter

from scipy.spatial.transform import Rotation as R
import sys

ros_path = "/opt/ros/kinetic/lib/python2.7/dist-packages"
if ros_path in sys.path:
    sys.path.remove(ros_path)
sys.path.append("../SuperGluePretrainedNetwork/")

import cv2

from matplotlib import pyplot as plt


import matplotlib.cm as cm
import torch
from mpl_toolkits import mplot3d

from models.matching import Matching
from models.utils import (
    AverageTimer,
    VideoStreamer,
    make_matching_plot_fast,
    frame2tensor,
)
import matplotlib
import weakref
import math

matplotlib.use("TkAgg")

from ekf import EKF

import copy
from transforms3d.euler import euler2mat
from agents.navigation.behavior_agent import BehaviorAgent


torch.set_grad_enabled(False)


# static things
sem_vals_allowed = [
    (70, 70, 70),
    (100, 40, 40),
    (153, 153, 153),
    (157, 234, 50),
    (128, 64, 128),
    (244, 35, 232),
    (102, 102, 156),
    (220, 220, 0),
    (250, 170, 30),
    (110, 190, 160),
    (81, 0, 81),
    (150, 100, 100),
    (230, 150, 140),
]


class SuperMatcher:

    resize = [640, 480]
    superglue = "outdoor"
    max_keypoints = -1
    nms_radius = 4
    keypoint_threshold = 0.1
    sinkhorn_iterations = 20
    match_threshold = 0.4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    show_keypoints = False

    config = {
        "superpoint": {
            "nms_radius": nms_radius,
            "keypoint_threshold": keypoint_threshold,
            "max_keypoints": max_keypoints,
        },
        "superglue": {
            "weights": superglue,
            "sinkhorn_iterations": sinkhorn_iterations,
            "match_threshold": match_threshold,
        },
    }

    def __init__(self):

        self.matching = Matching(self.config).eval().to(self.device)
        self.keys = ["keypoints", "scores", "descriptors"]

        ### Anchor Frame and associated data
        self.anchor_frame_tensor = None
        self.anchor_data = None
        self.anchor_frame = None
        self.anchor_image_id = None

    def set_anchor(self, frame):

        # Frame will be divided by 255
        self.anchor_frame_tensor = frame2tensor(frame, self.device)
        self.anchor_data = self.matching.superpoint({"image": self.anchor_frame_tensor})
        self.anchor_data = {k + "0": self.anchor_data[k] for k in self.keys}
        self.anchor_data["image0"] = self.anchor_frame_tensor
        self.anchor_frame = frame
        self.anchor_image_id = 0

    def process(self, frame):

        if self.anchor_frame_tensor is None:
            print("Please set anchor frame first...")
            return None

        frame_tensor = frame2tensor(frame, self.device)
        pred = self.matching({**self.anchor_data, "image1": frame_tensor})
        kpts0 = self.anchor_data["keypoints0"][0].cpu().numpy()
        kpts1 = pred["keypoints1"][0].cpu().numpy()
        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].cpu().numpy()

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])

        text = [
            "SuperGlue",
            "Keypoints: {}:{}".format(len(kpts0), len(kpts1)),
            "Matches: {}".format(len(mkpts0)),
        ]
        k_thresh = self.matching.superpoint.config["keypoint_threshold"]
        m_thresh = self.matching.superglue.config["match_threshold"]
        small_text = [
            "Keypoint Threshold: {:.4f}".format(k_thresh),
            "Match Threshold: {:.2f}".format(m_thresh),
        ]

        out = make_matching_plot_fast(
            self.anchor_frame,
            frame,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            path=None,
            show_keypoints=self.show_keypoints,
            small_text=small_text,
        )

        return out, mkpts0, mkpts1


class MotionEstimator:
    def estimate_R_t(self, pts1, pts2, intrinsics, depthMap, semanticMap):

        R = np.eye(3)
        t = np.zeros((3, 1))

        object_points = []
        pts2_filtered = []

        for i in range(len(pts1)):

            u1, v1 = pts1[i]

            if tuple(semanticMap[int(v1), int(u1)][:3][::-1]) in sem_vals_allowed:

                s = depthMap[int(v1), int(u1)] * 1000

                if s < 300:

                    pt = np.linalg.inv(intrinsics) @ (s * np.array([u1, v1, 1]))
                    object_points.append(pt)
                    pts2_filtered.append(pts2[i])

        object_points = np.vstack(object_points)
        pts2_filtered = np.vstack(pts2_filtered)

        _, rvec, t, inliers = cv2.solvePnPRansac(
            object_points, pts2_filtered, intrinsics, None
        )

        R, _ = cv2.Rodrigues(rvec)

        return R, t


class IMUSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find("sensor.other.imu")
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data)
        )
        self.timestamp = 0.0

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = np.array(
            (
                max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
                max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
                max(limits[0], min(limits[1], sensor_data.accelerometer.z)),
            )
        )
        self.gyroscope = np.array(
            (
                max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
                max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
                max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))),
            )
        )
        self.compass = math.degrees(sensor_data.compass)
        self.timestamp = sensor_data.timestamp


class Car:

    im_width = 640
    im_height = 480
    fov = 110

    actor_list = []

    front_camera = None
    front_camera_intrinsics = None
    front_camera_depth = None
    front_camera_depth_old = None
    front_camera_semantic = None
    front_camera_semantic_old = None

    imu_sensor = None

    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(2.0)

        self.world = self.client.get_world()

        blueprint_library = self.world.get_blueprint_library()
        self.model_3 = blueprint_library.filter("model3")[0]
        self.model_3.set_attribute('role_name', 'hero')
        self.agent = None
        
        settings = self.world.get_settings()
        settings.synchronous_mode = False  # Disables synchronous mode
        settings.fixed_delta_seconds = 0
        self.world.apply_settings(settings)
            

    def reset(self):
        
        actors = self.world.get_actors()
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in actors if "vehicle" in x.type_id]
        )
        self.client.apply_batch(
            [carla.command.DestroyActor(x) for x in actors if "sensor" in x.type_id]
        )
        for a in actors.filter("vehicle*"):
            if a.is_alive:
                a.destroy()
        for a in actors.filter("sensor*"):
            if a.is_alive:
                a.destroy()
                
        print("Scene init done!")
        
        self.actor_list = []


        self.transform = self.world.get_map().get_spawn_points()[0]
        self.transform.location.z += 1
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        # self.vehicle.set_autopilot()
        ##### Setup Agent #####
        self.agent = BehaviorAgent(self.vehicle, behavior="normal")
        destination_location = self.world.get_map().get_spawn_points()[50].location
        self.agent.set_destination(self.vehicle.get_location(), destination_location, clean=True)
        self.agent.update_information(self.world)
        
        
        ########################
        
        
        self.actor_list.append(self.vehicle)

        self.rgb_cam = self.world.get_blueprint_library().find("sensor.camera.rgb")

        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", "110")
        fx = self.im_width / (2 * np.tan(self.fov * np.pi / 360))
        fy = self.im_height / (2 * np.tan(self.fov * np.pi / 360))
        self.front_camera_intrinsics = np.array(
            [[fx, 0, self.im_width / 2], [0, fy, self.im_height / 2], [0, 0, 1]]
        )

        self.depth_cam = self.world.get_blueprint_library().find("sensor.camera.depth")
        self.depth_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.depth_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.depth_cam.set_attribute("fov", "110")

        self.semantic_cam = self.world.get_blueprint_library().find(
            "sensor.camera.depth"
        )
        self.semantic_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.semantic_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.semantic_cam.set_attribute("fov", "110")

        transform = carla.Transform(carla.Location(x=2.5, z=1))

        self.sensor_rgb = self.world.spawn_actor(
            self.rgb_cam, transform, attach_to=self.vehicle
        )
        self.actor_list.append(self.sensor_rgb)
        self.sensor_rgb.listen(lambda data: self.process_img(data))

        self.sensor_depth = self.world.spawn_actor(
            self.depth_cam, transform, attach_to=self.vehicle
        )
        self.actor_list.append(self.sensor_depth)
        self.sensor_depth.listen(lambda data: self.process_img_depth(data))

        self.sensor_semantic = self.world.spawn_actor(
            self.semantic_cam, transform, attach_to=self.vehicle
        )
        self.actor_list.append(self.sensor_semantic)
        self.sensor_semantic.listen(lambda data: self.process_img_semantic(data))

        self.imu_sensor = IMUSensor(self.vehicle)

        while self.front_camera is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        return self.front_camera

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        self.front_camera = i3

    def process_img_depth(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i2 = i2[:, :, :3]
        i3 = np.add(
            np.add(i2[:, :, 2], np.multiply(i2[:, :, 1], 256)),
            np.multiply(i2[:, :, 0], 256 * 256),
        )
        i3 = np.divide(i3, 256 ** 3 - 1)
        self.front_camera_depth_old = self.front_camera_depth
        self.front_camera_depth = i3

    def process_img_semantic(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))

        self.front_camera_semantic_old = self.front_camera_semantic
        self.front_camera_semantic = i2


def carla_rotation_to_RPY(carla_rotation):
    """
    Convert a carla rotation to a roll, pitch, yaw tuple
    Considers the conversion from left-handed system (unreal) to right-handed
    system (ROS).
    Considers the conversion from degrees (carla) to radians (ROS).
    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a tuple with 3 elements (roll, pitch, yaw)
    :rtype: tuple
    """
    roll = math.radians(carla_rotation.roll)
    pitch = -math.radians(carla_rotation.pitch)
    yaw = -math.radians(carla_rotation.yaw)

    return (roll, pitch, yaw)


def carla_rotation_to_numpy_rotation_matrix(carla_rotation):
    """
    Convert a carla rotation to a ROS quaternion
    Considers the conversion from left-handed system (unreal) to right-handed
    system (ROS).
    Considers the conversion from degrees (carla) to radians (ROS).
    :param carla_rotation: the carla rotation
    :type carla_rotation: carla.Rotation
    :return: a numpy.array with 3x3 elements
    :rtype: numpy.array
    """
    roll, pitch, yaw = carla_rotation_to_RPY(carla_rotation)
    numpy_array = euler2mat(roll, pitch, yaw)
    rotation_matrix = numpy_array[:3, :3]
    return rotation_matrix


if __name__ == "__main__":

    # Initial Setup #########################################################
    ## Initialize superglue + superpoint system
    superMatcher = SuperMatcher()

    ## Initialize PnP system
    motionEstimator = MotionEstimator()

    ## Create vehicle, and initialize the vehicle system
    vehicle = Car()
    vehicle.reset()

    ## Sleep for 2 seconds due to CARLA reasons (car needs some time to properly spawn)
    time.sleep(2)
    
    settings = vehicle.world.get_settings()
    settings.synchronous_mode = True  # Enables synchronous mode
    settings.fixed_delta_seconds = 0.1#0.025
    vehicle.world.apply_settings(settings)

    ## Initialize anchor, time reference,
    superMatcher.set_anchor(vehicle.front_camera[:, :, 0])
    t_prev = vehicle.imu_sensor.timestamp

    # Initialize plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")



    ### Location transforms ###############################################
    # we have 3 frames to deal with
    #   1. global CARLA frame
    #   2. robot start frame
    #   3. VO frame

    ## Get initial rotation and location in CARLA global frame (Left - Handed)
    initial_transform = vehicle.actor_list[0].get_transform()
    initial_rotation = initial_transform.rotation
    initial_location = initial_transform.location

    ## Convert global rotation and location to right-handed
    r_right_handed = R.from_matrix(
        carla_rotation_to_numpy_rotation_matrix(initial_rotation)
    )

    t_right_handed = np.array(
        [initial_location.x, initial_location.y * -1, initial_location.z]
    ).T

    # transform from start frame to global frame
    H_start_to_global = np.eye(4)
    H_start_to_global[:3, :3] = r_right_handed.as_matrix()
    H_start_to_global[:3, 3] = t_right_handed

    # transform from global frame to start frame
    H_global_to_start = np.linalg.inv(H_start_to_global)

    # handles difference between VO axes and CARLA axes. only gets applied at the end
    vo_compensation = (R.from_euler('x',-90,degrees=True) * R.from_euler('z',90,degrees=True)).as_matrix()
    


    # Initial trajectory points ##################################################
    ## Initialize container for robot poses, and trajectory (both in robot start frame)
    all_poses = [np.eye(4)]
    trajectory_vo = [np.array([0, 0, 0])]

    # initialize first ground truth position
    # transform from global frame to robot start frame
    initial_pos = H_global_to_start @ np.append(t_right_handed,1)
    trajectory_gt = [initial_pos]



    # EKF Initialization #TODO #########################################################
    # Convert the right-handed rotation to a quaternion, roll it to get the form w,x,y,z from x,y,z,w
    initial_quat_wxyz = np.roll(r_right_handed.as_quat(), 1)

    ## Initialize the EKF system #TODO check initial values
    vio_ekf = EKF(np.array([0,0,0]), np.array([0, 0, 0]), initial_quat_wxyz, debug=False)
    vio_ekf.use_new_data = False
    vio_ekf.setSigmaAccel(0.0)
    vio_ekf.setSigmaGyro(0.0)

    

    # Main Loop ############################################################# 
    first = True
    while True:
        # vehicle.vehicle.apply_control(vehicle.agent.run_step())
        # continue

        #### EKF Prediction #TODO #######################
        accel = copy.deepcopy(vehicle.imu_sensor.accelerometer)
        gyro = copy.deepcopy(vehicle.imu_sensor.gyroscope)
        t = copy.deepcopy(vehicle.imu_sensor.timestamp)

        # convert to right-handed coordinates 
        accel[1] *= -1
        gyro[1] *= -1

        # Perform prediction based on IMU signal
        vio_ekf.IMUPrediction(accel, gyro, t - t_prev)
        t_prev = t
        ###############################################
        
        

        # Visual Odometry ###########################################
        out_image_pair, pts1, pts2 = superMatcher.process(vehicle.front_camera[:, :, 0])
        R_, t = motionEstimator.estimate_R_t(
            pts1,
            pts2,
            vehicle.front_camera_intrinsics,
            vehicle.front_camera_depth_old,
            vehicle.front_camera_semantic_old,
        )
        current_pose = np.eye(4)
        current_pose[:3, :3] = R_
        current_pose[:3, 3] = t.reshape(
            3,
        )

        cv2.imshow("matches", out_image_pair)
        cv2.waitKey(1)
        superMatcher.set_anchor(vehicle.front_camera[:, :, 0])

        # new VO trajectory point is relative transformation from previous pose
        robot_pose_start = all_poses[-1] @ current_pose
        all_poses.append(robot_pose_start)
        position_start = robot_pose_start @ np.array([0, 0, 0, 1])
        trajectory_vo.append(position_start[:3])
        ###############################################################


        # EKF UPDATE #TODO #########################
        # vio_ekf.SuperGlueUpdate(position_xyz[:3])
        vio_ekf.addToStateList()
        ########################################


        # Add gt trajectory point (change to right handed coordinates,
        # then transform to robot start frame)
        gt_pos = vehicle.actor_list[0].get_location()
        gt_pos = np.array([gt_pos.x, -gt_pos.y, gt_pos.z, 1])
        trajectory_gt.append(H_global_to_start @ gt_pos)

        # Apply control on vehicle
        vehicle.vehicle.apply_control(vehicle.agent.run_step())
        vehicle.world.tick()


        if len(trajectory_vo) == 1000:
            # if len(trajectory_vo) == 25:
            break


        # Plotting ####################################

        # rotate VO points from VO frame to robot start frame
        trajectory_vo_np = (np.asarray(trajectory_vo) @ vo_compensation).T
        
        trajectory_gt_np = np.asarray(trajectory_gt).T
        trajectory_vio_np = vio_ekf.getTrajectory().T

        
        ax.plot3D(
            trajectory_gt_np[0],
            trajectory_gt_np[1],
            trajectory_gt_np[2],
            "red",
            label="Ground Truth",
        )

        ax.plot3D(
            trajectory_vo_np[0],
            trajectory_vo_np[1],
            trajectory_vo_np[2],
            "green",
            label="Estimated (VO)",
        )

        # ax.plot3D(
        #     trajectory_vio_np[0],
        #     trajectory_vio_np[1],
        #     trajectory_vio_np[2],
        #     "blue",
        #     label="Estimated (VIO)",
        # )

        
        # figure setup
        if first:
            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.set_zlabel("Z axis")
            ax.set_xlim3d(-200, 200)
            ax.set_ylim3d(-200, 200)
            ax.set_zlim3d(-200, 200)
            ax.legend()
            first = False

        plt.pause(0.05)

    for actor in vehicle.actor_list:
        actor.destroy()
    plt.show()
