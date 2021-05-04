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

import os

from datetime import datetime

torch.set_grad_enabled(False)

import gtsam
from gtsam.symbol_shorthand import B, V, X, L


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

        keypoints = self.anchor_data['keypoints0'][0].cpu().numpy()
        scores = self.anchor_data['scores0'][0].cpu().numpy()
        descriptors = self.anchor_data['descriptors0'][0].cpu().numpy()

        # N x 3
        pts = np.hstack((keypoints,scores.reshape(-1,1)))
        
        # N x 256 (256 is descriptor dimension)
        descriptors = descriptors.T



        return pts, descriptors

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
        # print("anchor",self.anchor_data.keys())
        # print(self.anchor_data['keypoints0'][0].shape)
        # print(self.anchor_data['scores0'][0].shape)
        # print(self.anchor_data['descriptors0'][0].shape)
        # print("pred",pred.keys()) 

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

class PointTracker(object):
  """ Class to manage a fixed memory of points and descriptors that enables
  sparse optical flow point tracking.

  Internally, the tracker stores a 'tracks' matrix sized M x (2+L), of M
  tracks with maximum length L, where each row corresponds to:
  row_m = [track_id_m, avg_desc_score_m, point_id_0_m, ..., point_id_L-1_m].
  """

  def __init__(self, max_length, nn_thresh):
    if max_length < 2:
      raise ValueError('max_length must be greater than or equal to 2.')
    self.maxl = max_length
    self.nn_thresh = nn_thresh
    self.all_pts = []
    for n in range(self.maxl):
      self.all_pts.append(np.zeros((2, 0)))
    self.last_desc = None
    self.tracks = np.zeros((0, self.maxl+2))
    self.track_count = 0
    self.max_score = 9999

  def nn_match_two_way(self, desc1, desc2, nn_thresh):
    """
    Performs two-way nearest neighbor matching of two sets of descriptors, such
    that the NN match from descriptor A->B must equal the NN match from B->A.

    Inputs:
      desc1 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      desc2 - NxM numpy matrix of N corresponding M-dimensional descriptors.
      nn_thresh - Optional descriptor distance below which is a good match.

    Returns:
      matches - 3xL numpy array, of L matches, where L <= N and each column i is
                a match of two descriptors, d_i in image 1 and d_j' in image 2:
                [d_i index, d_j' index, match_score]^T
    """
    assert desc1.shape[0] == desc2.shape[0]
    if desc1.shape[1] == 0 or desc2.shape[1] == 0:
      return np.zeros((3, 0))
    if nn_thresh < 0.0:
      raise ValueError('\'nn_thresh\' should be non-negative')
    # Compute L2 distance. Easy since vectors are unit normalized.
    dmat = np.dot(desc1.T, desc2)
    dmat = np.sqrt(2-2*np.clip(dmat, -1, 1))
    # Get NN indices and scores.
    idx = np.argmin(dmat, axis=1)
    scores = dmat[np.arange(dmat.shape[0]), idx]
    # Threshold the NN matches.
    keep = scores < nn_thresh
    # Check if nearest neighbor goes both directions and keep those.
    idx2 = np.argmin(dmat, axis=0)
    keep_bi = np.arange(len(idx)) == idx2[idx]
    keep = np.logical_and(keep, keep_bi)
    idx = idx[keep]
    scores = scores[keep]
    # Get the surviving point indices.
    m_idx1 = np.arange(desc1.shape[1])[keep]
    m_idx2 = idx
    # Populate the final 3xN match data structure.
    matches = np.zeros((3, int(keep.sum())))
    matches[0, :] = m_idx1
    matches[1, :] = m_idx2
    matches[2, :] = scores
    return matches

  def get_offsets(self):
    """ Iterate through list of points and accumulate an offset value. Used to
    index the global point IDs into the list of points.

    Returns
      offsets - N length array with integer offset locations.
    """
    # Compute id offsets.
    offsets = []
    offsets.append(0)
    for i in range(len(self.all_pts)-1): # Skip last camera size, not needed.
      offsets.append(self.all_pts[i].shape[1])
    offsets = np.array(offsets)
    offsets = np.cumsum(offsets)
    return offsets

  def update(self, pts, desc):
    """ Add a new set of point and descriptor observations to the tracker.

    Inputs
      pts - 3xN numpy array of 2D point observations.
      desc - DxN numpy array of corresponding D dimensional descriptors.
    """
    if pts is None or desc is None:
      print('PointTracker: Warning, no points were added to tracker.')
      return
    assert pts.shape[1] == desc.shape[1]
    # Initialize last_desc.
    if self.last_desc is None:
      self.last_desc = np.zeros((desc.shape[0], 0))
    # Remove oldest points, store its size to update ids later.
    remove_size = self.all_pts[0].shape[1]
    self.all_pts.pop(0)
    self.all_pts.append(pts)
    # Remove oldest point in track.
    self.tracks = np.delete(self.tracks, 2, axis=1)
    # Update track offsets.
    for i in range(2, self.tracks.shape[1]):
      self.tracks[:, i] -= remove_size
    self.tracks[:, 2:][self.tracks[:, 2:] < -1] = -1
    offsets = self.get_offsets()
    # Add a new -1 column.
    self.tracks = np.hstack((self.tracks, -1*np.ones((self.tracks.shape[0], 1))))
    # Try to append to existing tracks.
    matched = np.zeros((pts.shape[1])).astype(bool)
    matches = self.nn_match_two_way(self.last_desc, desc, self.nn_thresh)
    for match in matches.T:
      # Add a new point to it's matched track.
      id1 = int(match[0]) + offsets[-2]
      id2 = int(match[1]) + offsets[-1]
      found = np.argwhere(self.tracks[:, -2] == id1)
      if found.shape[0] > 0:
        matched[int(match[1])] = True
        row = int(found)
        self.tracks[row, -1] = id2
        if self.tracks[row, 1] == self.max_score:
          # Initialize track score.
          self.tracks[row, 1] = match[2]
        else:
          # Update track score with running average.
          # NOTE(dd): this running average can contain scores from old matches
          #           not contained in last max_length track points.
          track_len = (self.tracks[row, 2:] != -1).sum() - 1.
          frac = 1. / float(track_len)
          self.tracks[row, 1] = (1.-frac)*self.tracks[row, 1] + frac*match[2]
    # Add unmatched tracks.
    new_ids = np.arange(pts.shape[1]) + offsets[-1]
    new_ids = new_ids[~matched]
    new_tracks = -1*np.ones((new_ids.shape[0], self.maxl + 2))
    new_tracks[:, -1] = new_ids
    new_num = new_ids.shape[0]
    new_trackids = self.track_count + np.arange(new_num)
    new_tracks[:, 0] = new_trackids
    new_tracks[:, 1] = self.max_score*np.ones(new_ids.shape[0])
    self.tracks = np.vstack((self.tracks, new_tracks))
    self.track_count += new_num # Update the track count.
    
    # Remove empty tracks.
    keep_rows = np.any(self.tracks[:, 2:] >= 0, axis=1)
    self.tracks = self.tracks[keep_rows, :]
    
    # Store the last descriptors.
    self.last_desc = desc.copy()
    return

  def get_tracks(self, min_length):
    """ Retrieve point tracks of a given minimum length.
    Input
      min_length - integer >= 1 with minimum track length
    Output
      returned_tracks - M x (2+L) sized matrix storing track indices, where
        M is the number of tracks and L is the maximum track length.
    """
    if min_length < 1:
      raise ValueError('\'min_length\' too small.')
    valid = np.ones((self.tracks.shape[0])).astype(bool)
    good_len = np.sum(self.tracks[:, 2:] != -1, axis=1) >= min_length
    # Remove tracks which do not have an observation in most recent frame.
    not_headless = (self.tracks[:, -1] != -1)
    keepers = np.logical_and.reduce((valid, good_len, not_headless))
    returned_tracks = self.tracks[keepers, :].copy()
    return returned_tracks

  def draw_tracks(self, out, tracks):
    """ Visualize tracks all overlayed on a single image.
    Inputs
      out - numpy uint8 image sized HxWx3 upon which tracks are overlayed.
      tracks - M x (2+L) sized matrix storing track info.
    """
    # Store the number of points per camera.
    pts_mem = self.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = self.get_offsets()
    # Width of track and point circles to be drawn.
    stroke = 1
    # Iterate through each track and draw it.
    for track in tracks:
      clr = myjet[int(np.clip(np.floor(track[1]*10), 0, 9)), :]*255
      for i in range(N-1):
        if track[i+2] == -1 or track[i+3] == -1:
          continue
        offset1 = offsets[i]
        offset2 = offsets[i+1]
        idx1 = int(track[i+2]-offset1)
        idx2 = int(track[i+3]-offset2)
        pt1 = pts_mem[i][:2, idx1]
        pt2 = pts_mem[i+1][:2, idx2]
        p1 = (int(round(pt1[0])), int(round(pt1[1])))
        p2 = (int(round(pt2[0])), int(round(pt2[1])))
        cv2.line(out, p1, p2, clr, thickness=stroke, lineType=16)
        # Draw end points of each track.
        if i == N-2:
          clr2 = (255, 0, 0)
          cv2.circle(out, p2, stroke, clr2, -1, lineType=16)


def get_vision_data(tracker):
    """ Get keypoint-data pairs from the tracks. 
    """
    # Store the number of points per camera.
    pts_mem = tracker.all_pts
    N = len(pts_mem) # Number of cameras/images.
    # Get offset ids needed to reference into pts_mem.
    offsets = tracker.get_offsets()
    # Iterate through each track and get the data from the current image.
    vision_data = -1 * np.ones((tracker.tracks.shape[0], N, 2), dtype=int)
    for j, track in enumerate(tracker.tracks):
      for i in range(N-1):
        if track[i+3] == -1: # track[i+2] == -1 or 
          continue
        offset2 = offsets[i+1]
        idx2 = int(track[i+3]-offset2)
        pt2 = pts_mem[i+1][:2, idx2]
        vision_data[j, i] = np.array([int(round(pt2[0])), int(round(pt2[1]))])
    return vision_data

class VisualInertialOdometryGraph(object):
    
    def __init__(self, intrinsics, IMU_PARAMS=None, BIAS_COVARIANCE=None):
        """
        Define factor graph parameters (e.g. noise, camera calibrations, etc) here
        """
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.IMU_PARAMS = IMU_PARAMS
        self.BIAS_COVARIANCE = BIAS_COVARIANCE
        self.K = intrinsics

    def add_imu_measurements(self, measured_poses, measured_acc, measured_omega, measured_vel, delta_t, n_skip, initial_poses=None):

        n_frames = measured_poses.shape[0]

        # Check if sizes are correct
        assert measured_poses.shape[0] == n_frames
        assert measured_acc.shape[0] == n_frames
        assert measured_vel.shape[0] == n_frames

        # Pose prior
        pose_key = X(0)
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))
        pose_0 = gtsam.Pose3(measured_poses[0])
        self.graph.push_back(gtsam.PriorFactorPose3(pose_key, pose_0, pose_noise))

        self.initial_estimate.insert(pose_key, gtsam.Pose3(measured_poses[0]))

        # IMU prior
        # bias_key = B(0)
        # bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 0.5)
        # self.graph.push_back(gtsam.PriorFactorConstantBias(bias_key, gtsam.imuBias.ConstantBias(), bias_noise))

        # self.initial_estimate.insert(bias_key, gtsam.imuBias.ConstantBias())

        # Velocity prior
        # velocity_key = V(0)
        # velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, .5)
        # velocity_0 = measured_vel[0]
        # self.graph.push_back(gtsam.PriorFactorVector(velocity_key, velocity_0, velocity_noise))

        # self.initial_estimate.insert(velocity_key, velocity_0)
        
        # Preintegrator
        accum = gtsam.PreintegratedImuMeasurements(self.IMU_PARAMS)

        # Add measurements to factor graph
        for i in range(1, n_frames):
            accum.integrateMeasurement(measured_acc[i], measured_omega[i], delta_t[i-1])
            if i % n_skip == 0:
                pose_key += 1
                DELTA = gtsam.Pose3(gtsam.Rot3.Rodrigues(0, 0, 0.1 * np.random.randn()),
                                    gtsam.Point3(4 * np.random.randn(), 4 * np.random.randn(), 4 * np.random.randn()))
                self.initial_estimate.insert(pose_key, gtsam.Pose3(measured_poses[i]).compose(DELTA))

                # velocity_key += 1
                # self.initial_estimate.insert(velocity_key, measured_vel[i])

                # bias_key += 1
                # self.graph.add(gtsam.BetweenFactorConstantBias(bias_key - 1, bias_key, gtsam.imuBias.ConstantBias(), self.BIAS_COVARIANCE))
                # self.initial_estimate.insert(bias_key, gtsam.imuBias.ConstantBias())

                # Add IMU Factor
                self.graph.add(gtsam.ImuFactor(pose_key - 1, velocity_key - 1, pose_key, velocity_key, bias_key, accum))

                # Reset preintegration
                accum.resetIntegration()

    def add_keypoints(self,vision_data,measured_poses,n_skip, depth):
        
      measured_poses = np.reshape(measured_poses,(-1,4,4))
      pose_key = X(0)
      self.initial_estimate.insert(pose_key, gtsam.Pose3(measured_poses[0]))
    #   R_rect = np.array([[9.999239e-01, 9.837760e-03, -7.445048e-03, 0.],
    #                      [ -9.869795e-03, 9.999421e-01, -4.278459e-03, 0.],
    #                      [ 7.402527e-03, 4.351614e-03, 9.999631e-01, 0.],
    #                      [ 0., 0., 0., 1.]])
    #   R_cam_velo = np.array([[7.533745e-03, -9.999714e-01, -6.166020e-04],
    #                          [ 1.480249e-02, 7.280733e-04, -9.998902e-01],
    #                          [ 9.998621e-01, 7.523790e-03, 1.480755e-02]])
    #   R_velo_imu = np.array([[9.999976e-01, 7.553071e-04, -2.035826e-03],
    #                          [-7.854027e-04, 9.998898e-01, -1.482298e-02],
    #                          [2.024406e-03, 1.482454e-02, 9.998881e-01]])
    #   t_cam_velo = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01])
    #   t_velo_imu = np.array([-8.086759e-01, 3.195559e-01, -7.997231e-01])
    #   T_velo_imu = np.zeros((4,4))
    #   T_cam_velo = np.zeros((4,4))
    #   T_velo_imu[3,3] = 1.
    #   T_cam_velo[3,3] = 1.
    #   T_velo_imu[:3,:3] = R_velo_imu
    #   T_velo_imu[:3,3] = t_velo_imu
    #   T_cam_velo[:3,:3] = R_cam_velo
    #   T_cam_velo[:3,3] = t_cam_velo
    #   cam_to_imu = R_rect @ T_cam_velo @ T_velo_imu
    #   CAM_TO_IMU_POSE = gtsam.Pose3(cam_to_imu)
    #   imu_to_cam = np.linalg.inv(cam_to_imu)
    #   IMU_TO_CAM_POSE = gtsam.Pose3(imu_to_cam)

    #   K_np = np.array([[9.895267e+02, 0.000000e+00, 7.020000e+02], 
    #                    [0.000000e+00, 9.878386e+02, 2.455590e+02], 
    #                    [0.000000e+00, 0.000000e+00, 1.000000e+00]]) 

      K_np = self.K

      K = gtsam.Cal3_S2(K_np[0,0], K_np[1,1], 0., K_np[0,2], K_np[1,2])

      valid_track = np.zeros(vision_data.shape[0], dtype=bool)
      N = vision_data.shape[1]
      for i in range(vision_data.shape[0]):
          track_length = N - np.sum(vision_data[i,:,0] == -1)
          if track_length > 1 and track_length < 0.5*N:
              valid_track[i] = True

      count = 0
      measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 10.0) 
      for i in range(0, vision_data.shape[0], 20):
        if not valid_track[i]:
            continue
        key_point_initialized=False 
        for j in range(vision_data.shape[1]-1):
          if vision_data[i,j,0] >= 0:
            zp = float(depth[j * n_skip][vision_data[i,j,1], vision_data[i,j,0]])
            if zp == 0:
                continue
            self.graph.push_back(gtsam.GenericProjectionFactorCal3_S2(
              vision_data[i,j,:], measurement_noise, X(j), L(i), K, gtsam.Pose3(np.eye(4))))
            if not key_point_initialized:
                count += 1

                # Initialize landmark 3D coordinates
                fx = K_np[0,0]
                fy = K_np[1,1]
                cx = K_np[0,2]
                cy = K_np[1,2]

                # Depth:
                zp = float(depth[j * n_skip][vision_data[i,j,1], vision_data[i,j,0]])
                xp = float(vision_data[i,j,0] - cx) / fx * zp
                yp = float(vision_data[i,j,1] - cy) / fy * zp

                # Convert to global
                Xg = measured_poses[j*n_skip] @ np.array([xp, yp, zp, 1])
                self.initial_estimate.insert(L(i), Xg[:3])
                
                key_point_initialized = True

      print('==> Using ', count, ' tracks')

    def estimate(self, SOLVER_PARAMS=None):
        self.optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial_estimate, SOLVER_PARAMS)
        self.result = self.optimizer.optimize()

        return self.result

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
    settings.fixed_delta_seconds = 0.1#0.025#0.025
    vehicle.world.apply_settings(settings)

    ## Initialize anchor, time reference, initial sensor readings
    pts, desc = superMatcher.set_anchor(vehicle.front_camera[:, :, 0])
    t_prev = vehicle.imu_sensor.timestamp
    accel = copy.deepcopy(vehicle.imu_sensor.accelerometer)
    gyro = copy.deepcopy(vehicle.imu_sensor.gyroscope)

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
    # initial_quat_wxyz = np.roll(R.from_matrix(car_start_r).as_quat(), 1)

    ## Initialize the EKF system #TODO check initial values
    # vio_ekf = EKF(np.array([0,0,0]), np.array([0, 0, 0]), initial_quat_wxyz, debug=False)
    vio_ekf = EKF(np.array([0,0,0]), np.array([0, 0, 0]), np.array([1,0,0,0]), debug=False)
    vio_ekf.use_new_data = True
    vio_ekf.setSigmaAccel(1.)
    vio_ekf.setSigmaGyro(0.5)
    vio_ekf.setSigmaVO(5.)

    

    accel_list = []
    gyro_list = []
    
    # Main Loop ############################################################# 
    max_length = 10
    first = True
    step = 0

    tracker = PointTracker(max_length=max_length, nn_thresh=0.9)

    depth_list = []

    while True:
        step += 1

        tracker.update(pts.T,desc.T)
        depth_list.append(vehicle.front_camera_depth_old)

        
        # vehicle.vehicle.apply_control(vehicle.agent.run_step())
        # continue

        #### EKF Prediction #TODO #######################
        t = copy.deepcopy(vehicle.imu_sensor.timestamp)
        next_accel = copy.deepcopy(vehicle.imu_sensor.accelerometer)
        next_gyro = copy.deepcopy(vehicle.imu_sensor.gyroscope)


        # convert to right-handed coordinates
        # accel[1] *= -1
        accel[2] *= -1
        
        gyro[1] *= -1
        # gyro[2] *= -1
        gyro *= np.pi / 180 # radians

        print(accel,"accel")
        print(gyro,"gyro")
        print()
        accel_list.append(accel)
        gyro_list.append(gyro)

        # Perform prediction based on IMU signal
        vio_ekf.IMUPrediction(accel, gyro, t - t_prev)
        t_prev = copy.deepcopy(t)
        accel = copy.deepcopy(next_accel)
        gyro = copy.deepcopy(next_gyro)
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
        pts,desc = superMatcher.set_anchor(vehicle.front_camera[:, :, 0])

        # new VO trajectory point is relative transformation from previous pose
        robot_pose_start = all_poses[-1] @ current_pose
        all_poses.append(robot_pose_start)
        position_start = robot_pose_start @ np.array([0, 0, 0, 1])
        trajectory_vo.append(position_start[:3])
        ###############################################################


        # EKF UPDATE #TODO #########################
        if(step % 10 == 0):
            # vio_ekf.SuperGlueUpdate((copy.deepcopy(position_start[:3]).T @ vo_compensation).T)
            vo_pose = (copy.deepcopy(position_start[:3]).T @ vo_compensation).T
            vo_pose[2] = 0
            vio_ekf.SuperGlueUpdate(vo_pose[:3])

        
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

        ax.plot3D(
            trajectory_vio_np[0],
            trajectory_vio_np[1],
            trajectory_vio_np[2],
            "blue",
            label="Estimated (VIO)",
        )

        
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

        if len(trajectory_vo) == max_length:
            break

    gyro_list_np = np.array(gyro_list)
    accel_list_np = np.array(accel_list)

    np.savez("imu_output.npz",accel = accel_list_np,gyro = gyro_list_np,gt=trajectory_gt_np)

    if not os.path.exists('output'):
        os.makedirs('output')

    now = datetime.now()
    npz_path = 'output/' + now.strftime("%d_%m_%H_%M_%S.npz")

    np.savez(npz_path,gt=trajectory_gt_np,vo=trajectory_vo_np,vio=trajectory_vio_np)

    ######################################################################################################
    vision_data = get_vision_data(tracker)

    g=9.81
    IMU_PARAMS = gtsam.PreintegrationParams.MakeSharedU(g)
    I = np.eye(3)
    IMU_PARAMS.setAccelerometerCovariance(I * 0.2)
    IMU_PARAMS.setGyroscopeCovariance(I * 0.2)
    IMU_PARAMS.setIntegrationCovariance(I * 0.2)

    BIAS_COVARIANCE = gtsam.noiseModel.Isotropic.Variance(6, 0.4)
    
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(1000)
    params.setlambdaUpperBound(1.e+6)
    params.setlambdaLowerBound(0.1)
    params.setDiagonalDamping(1000)
    params.setVerbosity('ERROR')
    params.setVerbosityLM('SUMMARY')
    params.setRelativeErrorTol(1.e-9)
    params.setAbsoluteErrorTol(1.e-9)

    vio_full = VisualInertialOdometryGraph(vehicle.front_camera_intrinsics,IMU_PARAMS=IMU_PARAMS, BIAS_COVARIANCE=BIAS_COVARIANCE)
    
    vio_full.add_keypoints(vision_data, vio_ekf.pose_list, 1, depth_list)

    result_full = vio_full.estimate(SOLVER_PARAMS=params)



    for actor in vehicle.actor_list:
        actor.destroy()
