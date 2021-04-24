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


torch.set_grad_enabled(False)

sem_vals_allowed = [
    (70, 70, 70),
    (100, 40, 40),
    (153, 153, 153),
    (157, 234, 50),
    (244, 35, 232),
    (102, 102, 156),
    (250, 170, 30),
    (110, 190, 160),
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

        _, rvec, t, inliners = cv2.solvePnPRansac(
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
        self.timestamp = 0.

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = np.array((
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)),
        ))
        self.gyroscope = np.array((
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))),
        ))
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

    def reset(self):
        self.actor_list = []

        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.transform.location.z += 1
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.vehicle.set_autopilot()
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
        # i = np.array(image.raw_data)
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))

        self.front_camera_semantic_old = self.front_camera_semantic
        self.front_camera_semantic = i2
        # brak


if __name__ == "__main__":

    superMatcher = SuperMatcher()
    motionEstimator = MotionEstimator()

    vehicle = Car()
    vehicle.reset()

    time.sleep(4)

    superMatcher.set_anchor(vehicle.front_camera[:, :, 0])
    t_prev = vehicle.imu_sensor.timestamp
    print(t_prev)
    counter = 0

    all_poses = [np.eye(4)]
    trajectory_vo = [np.array([0, 0, 0])]

    initial_transform = vehicle.actor_list[0].get_transform()
    initial_rotation = initial_transform.rotation
    initial_location = initial_transform.location

    r_right_handed = R.from_rotvec(
        -np.pi
        / 180
        * np.array(
            [initial_rotation.roll, initial_rotation.pitch, initial_rotation.yaw]
        )
    )
    t_right_handed = np.array(
        [initial_location.x, initial_location.y * -1, initial_location.z]
    ).T

    H_local_to_global = np.eye(4)
    H_local_to_global[:3, :3] = r_right_handed.as_matrix()
    H_local_to_global[:3, 3] = t_right_handed
    H_global_to_local = np.linalg.inv(H_local_to_global)

    r_right_handed = R.from_matrix(H_global_to_local[:3, :3])
    t_right_handed = H_global_to_local[:3, 3]

    r_right_handed = r_right_handed.as_rotvec()
    r_left_handed = -1 * r_right_handed * 180 / np.pi
    t_left_handed = t_right_handed.T
    t_left_handed[1] = t_left_handed[1] * -1

    roll, pitch, yaw = r_left_handed
    x, y, z = t_left_handed

    inv_transform = carla.Transform(
        location=carla.Location(x=x, y=y, z=z),
        rotation=carla.Rotation(roll=roll, pitch=pitch, yaw=yaw),
    )

    inv_transform.transform(initial_location)

    initial_pos = np.array([initial_location.x, initial_location.y, initial_location.z])

    trajectory_gt = [initial_pos]

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    vio_ekf = EKF(np.array([0,0,0]),np.array([0,0,0]),np.array([1,0,0,0]),debug=False)

    first = True
    while True:

        #### EKF Prediction #######################
        accel = vehicle.imu_sensor.accelerometer
        gyro = vehicle.imu_sensor.gyroscope
        t = vehicle.imu_sensor.timestamp

        vio_ekf.IMUPrediction(accel,gyro,t-t_prev)
        t_prev = t
        ########################################3

        out_image_pair, pts1, pts2 = superMatcher.process(vehicle.front_camera[:, :, 0])
        R, t = motionEstimator.estimate_R_t(
            pts1,
            pts2,
            vehicle.front_camera_intrinsics,
            vehicle.front_camera_depth_old,
            vehicle.front_camera_semantic_old,
        )
        current_pose = np.eye(4)
        current_pose[:3, :3] = R
        current_pose[:3, 3] = t.reshape(
            3,
        )

        global_robot_pose = all_poses[-1] @ current_pose
        all_poses.append(global_robot_pose)
        position_xyz = global_robot_pose @ np.array([0, 0, 0, 1])
        trajectory_vo.append(position_xyz[:3])

        # EKF UPDATE #########################
        vio_ekf.SuperGlueUpdate(position_xyz[:3],use_new_data=False)
        vio_ekf.addToStateList()
         ########################################

        print("Trajectory Length:", len(trajectory_vo))

        gt_pos = vehicle.actor_list[0].get_location()

        # gt_pos -= gt_origin
        gt_pos = inv_transform.transform(gt_pos)
        gt_pos = np.array([gt_pos.x, gt_pos.y, gt_pos.z])

        trajectory_gt.append(gt_pos)

        cv2.imshow("matches", out_image_pair)
        # cv2.imshow("depth", vehicle.front_camera_depth_old)
        cv2.waitKey(1)
        superMatcher.set_anchor(vehicle.front_camera[:, :, 0])

    
        if len(trajectory_vo) == 1000:
        # if len(trajectory_vo) == 25:
            break
        trajectory_vo_np = np.asarray(trajectory_vo).T
        trajectory_gt_np = np.asarray(trajectory_gt).T
        trajectory_vio_np = vio_ekf.getTrajectory().T

        ax.plot3D(-1 * trajectory_vo_np[2], trajectory_vo_np[0], trajectory_vo_np[1], "green",label = "Estimated (VO)") 
        ax.plot3D(-1 * trajectory_vio_np[2], trajectory_vio_np[0], trajectory_vio_np[1], "blue",label="Estimated (VIO)")
        ax.plot3D(trajectory_gt_np[0], trajectory_gt_np[1], trajectory_gt_np[2], "red",label="Ground Truth")

        if first:
            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.set_zlabel("Z axis")
            ax.set_xlim3d(-50, 50)
            ax.set_ylim3d(-50, 50)
            ax.set_zlim3d(-50, 50)
            ax.legend()
            first = False

        plt.pause(0.05)

    for actor in vehicle.actor_list:
        actor.destroy()
    plt.show()
