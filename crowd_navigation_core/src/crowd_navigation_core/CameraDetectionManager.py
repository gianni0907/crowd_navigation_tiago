import numpy as np
import math
import time
import os
import json
import threading
import cProfile
import tf2_ros
import rospy
import cv2
import torch
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
from image_geometry import PinholeCameraModel as PC

from crowd_navigation_core.Hparams import *
from crowd_navigation_core.utils import *

import gazebo_msgs.msg
import sensor_msgs.msg
import crowd_navigation_msgs.msg

class CameraDetectionManager:
    '''
    Extract the agents' representative 2D-point from the RGB-D camera
    '''
    def __init__(self):
        self.data_lock = threading.Lock()

        # Set status, two possibilities:
        # WAITING: waiting for the initial robot configuration
        # READY: reading from the camera
        self.status = Status.WAITING
        
        self.robot_config = Configuration(0.0, 0.0, 0.0)
        self.hparams = Hparams()
        self.bridge = CvBridge()
        self.rgb_image_nonrt = None
        self.depth_image_nonrt = None
        self.model = YOLO("yolov8n.pt")
        if self.hparams.simulation:
            self.agents_pos_nonrt = np.zeros((self.hparams.n_agents, 2))
            self.agents_name = ['actor_{}'.format(i) for i in range(self.hparams.n_agents)]

        # Set variables to store data
        if self.hparams.log:
            self.time_history = []
            self.measurements_history = []
            self.robot_config_history = []
            self.camera_pos_history = []
            self.camera_pan_history = []
            self.boundary_vertexes = []
            if self.hparams.simulation:
                self.agents_pos_history = []

        # Setup reference frames
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'
        self.camera_frame = 'xtion_rgb_optical_frame'

        # Setup TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        if self.hparams.simulation:
            # Setup subscriber to image_raw topic
            image_topic = '/xtion/rgb/image_raw'
            rospy.Subscriber(
                image_topic,
                sensor_msgs.msg.Image,
                self.image_callback
            )
            # Setup subscriber to depth image topic
            depth_topic = '/xtion/depth_registered/image_raw'
            rospy.Subscriber(
                depth_topic,
                sensor_msgs.msg.Image,
                self.depth_callback
            )
        else:
            # Setup subscriber to image_raw/compressed topic
            image_topic = '/xtion/rgb/image_raw/compressed'
            rospy.Subscriber(
                image_topic,
                sensor_msgs.msg.CompressedImage,
                self.compressed_image_callback
            )
            # Setup subscriber to depth image topic
            depth_topic = '/xtion/depth_registered/image/compressed'
            rospy.Subscriber(
                depth_topic,
                sensor_msgs.msg.CompressedImage,
                self.compressed_depth_callback
            )

        # Setup publisher to the image topic
        processed_image_topic = '/image'
        self.processed_image_publisher = rospy.Publisher(
            processed_image_topic,
            sensor_msgs.msg.Image,
            queue_size=1
        )

        # Setup subscriber to model_states topic
        model_states_topic = "/gazebo/model_states"
        rospy.Subscriber(
            model_states_topic,
            gazebo_msgs.msg.ModelStates,
            self.gazebo_model_states_callback
        )

        # Setup publisher to camera_measurements topic
        measurements_topic = 'camera_measurements'
        self.measurements_publisher = rospy.Publisher(
            measurements_topic,
            crowd_navigation_msgs.msg.MeasurementsStamped,
            queue_size=1
        )

    def image_callback(self, data):
        try:
            self.rgb_image_nonrt = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            rospy.logerr(e)

    def compressed_image_callback(self, data):
        try:
            self.rgb_image_nonrt = self.bridge.compressed_imgmsg_to_cv2(data, 'bgr8')
        except Exception as e:
            rospy.logerr(e)

    def depth_callback(self, data):
        try:
            self.depth_image_nonrt = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr(e)

    def compressed_depth_callback(self, data):
        try:
            self.depth_image_nonrt = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding='passthrough')
        except Exception as e:
            rospy.logerr(e)

    def gazebo_model_states_callback(self, msg):
        if self.hparams.simulation:
            agents_pos = np.zeros((self.hparams.n_agents, 2))
            idx = 0
            for agent_name in self.agents_name:
                if agent_name in msg.name:
                    agent_idx = msg.name.index(agent_name)
                    p = msg.pose[agent_idx].position
                    agent_pos = np.array([p.x, p.y])
                    agents_pos[idx] = agent_pos
                    idx += 1
            self.agents_pos_nonrt = agents_pos

    def tf2q(self, transform):
        q = transform.transform.rotation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.robot_config.theta = theta
        self.robot_config.x = transform.transform.translation.x + self.hparams.b * math.cos(theta)
        self.robot_config.y = transform.transform.translation.y + self.hparams.b * math.sin(theta)

    def update_configuration(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_footprint_frame, rospy.Time()
            )
            self.tf2q(transform)
            return True
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing current configuration")
            return False

    def get_camera_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.camera_frame, rospy.Time()
            )
            q = np.array([transform.transform.rotation.x,
                          transform.transform.rotation.y,
                          transform.transform.rotation.z,
                          transform.transform.rotation.w])
            
            self.pan_angle = math.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                                        1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
            
            pos = np.array([transform.transform.translation.x, 
                            transform.transform.translation.y, 
                            transform.transform.translation.z])
            
            rotation_matrix = R.from_quat(q).as_matrix()
            self.camera_pose = np.eye(4)
            self.camera_pose[:3, :3] = rotation_matrix
            self.camera_pose[:3, 3] = pos
            return True
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing camera pose")
            return False

    def data_extraction(self, rgb_img, depth_img):
        robot_position = self.robot_config.get_q()[:2]
        core_points = []
        results = self.model(rgb_img, iou=0.5, conf=0.5, verbose=False)
        for result in results:
            labels, cords = result.boxes.cls, result.boxes.xyxy.cpu().numpy()
            for label, cord in zip(labels, cords):
                if label == 0:
                    box_center = np.array([(cord[0] + cord[2]) / 2,
                                           (cord[1] + cord[3]) / 2])
                    x_min = int(cord[0])
                    y_min = int(cord[1])
                    x_max = int(cord[2])
                    y_max = int(cord[3])
                    # depth_min = np.nanmin(depth_img[y_min: y_max + 1, x_min: x_max + 1])
                    # depth_max = np.nanmax(depth_img[y_min: y_max + 1, x_min: x_max + 1])
                    # depth_mean = np.nanmean(depth_img[y_min: y_max + 1, x_min: x_max + 1])
                    # depth_median = np.nanmedian(depth_img[y_min: y_max + 1, x_min: x_max + 1])
                    depth= np.nanpercentile(depth_img[y_min: y_max + 1, x_min: x_max + 1], 20)
                    # print(f"Bbox center: {box_center}")
                    # print(f"depth: {depth}")
                    # print(f"depth min: {depth_min}")
                    # print(f"depth max: {depth_max}")
                    # print(f"depth mean: {depth_mean}")
                    # print(f"depth median: {depth_median}")
                    if depth >= self.hparams.cam_min_range:
                        point_cam = np.array(self.imgproc.projectPixelTo3dRay(box_center))
                        scale = depth / point_cam[2]
                        point_cam = point_cam * scale
                        homo_point_cam = np.append(point_cam, 1)
                        point_wrld = np.matmul(self.camera_pose, homo_point_cam)[:2]
                        if not is_outside(point_wrld, self.hparams.vertexes, self.hparams.normals):
                            core_points.append(point_wrld)
            processed_img = result.plot()
            self.processed_image_publisher.publish(self.bridge.cv2_to_imgmsg(processed_img))

        core_points = np.array(core_points)
        core_points, _ = sort_by_distance(core_points, robot_position)
        core_points = core_points[:self.hparams.n_filters]

        return core_points, processed_img

    def log_values(self):
        output_dict = {}
        output_dict['cpu_time'] = self.time_history
        output_dict['measurements'] = self.measurements_history
        output_dict['robot_config'] = self.robot_config_history
        output_dict['camera_position'] = self.camera_pos_history
        output_dict['camera_pan'] = self.camera_pan_history
        output_dict['frequency'] = self.hparams.camera_detector_frequency
        output_dict['b'] = self.hparams.b
        output_dict['n_filters'] = self.hparams.n_filters
        output_dict['n_points'] = self.hparams.n_points
        output_dict['min_range'] = self.hparams.cam_min_range
        output_dict['max_range'] = self.hparams.cam_max_range
        output_dict['horz_fov'] = self.hparams.cam_horz_fov
        for i in range(self.hparams.n_points):
            self.boundary_vertexes.append(self.hparams.vertexes[i].tolist())
        output_dict['boundary_vertexes'] = self.boundary_vertexes
        output_dict['base_radius'] = self.hparams.base_radius
        output_dict['simulation'] = self.hparams.simulation
        if self.hparams.simulation:
            output_dict['n_agents'] = self.hparams.n_agents
            output_dict['agents_pos'] = self.agents_pos_history
            output_dict['agent_radius'] = self.hparams.ds_cbf

        # log the data in a .json file
        log_dir = self.hparams.log_dir
        filename = self.hparams.camera_detector_file
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, filename)
        with open(log_path, 'w') as file:
            json.dump(output_dict, file)

        self.out.release()

    def run(self):
        cam_info = rospy.wait_for_message("/xtion/rgb/camera_info", sensor_msgs.msg.CameraInfo, timeout=None)
        self.imgproc = PC()
        self.imgproc.fromCameraInfo(cam_info)

        rate = rospy.Rate(self.hparams.camera_detector_frequency)

        if self.hparams.n_filters == 0:
            rospy.logwarn("No agent considered, camera detection disabled")
            return

        if self.hparams.perception == Perception.FAKE or self.hparams.perception == Perception.LASER:
            rospy.logwarn("Camera detection disabled")
            return

        if self.hparams.log:
            # Setup variables to create a video
            self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            self.out = cv2.VideoWriter(os.path.join(self.hparams.log_dir, self.hparams.filename + '_camera_view.mp4'),
                                       self.fourcc,
                                       self.hparams.camera_detector_frequency,
                                       (cam_info.width, cam_info.height))
            rospy.on_shutdown(self.log_values)

        while not rospy.is_shutdown():
            start_time = time.time()

            if self.status == Status.WAITING:
                if self.update_configuration() and self.get_camera_pose():
                    self.status = Status.READY
                else:
                    rate.sleep()
                    continue

            if self.rgb_image_nonrt is None or self.depth_image_nonrt is None:
                rospy.logwarn("Missing camera data")
                rate.sleep()
                continue

            with self.data_lock:
                self.update_configuration()
                self.get_camera_pose()
                rgb_image = self.rgb_image_nonrt
                depth_image = self.depth_image_nonrt
                if self.hparams.simulation:
                    agents_pos = self.agents_pos_nonrt

            measurements, processed_image = self.data_extraction(rgb_image, depth_image)

            # Create measurements message
            measurements_obj = Measurements()
            for measurement in measurements:
                measurements_obj.append(Position(measurement[0], measurement[1]))
            measurements_stamped = MeasurementsStamped(rospy.Time.from_sec(start_time),
                                                       'map',
                                                       measurements_obj)
            measurements_stamped_msg = MeasurementsStamped.to_message(measurements_stamped)
            self.measurements_publisher.publish(measurements_stamped_msg)

            # Update logged data
            if self.hparams.log:
                self.out.write(processed_image)
                self.robot_config_history.append([self.robot_config.x,
                                                  self.robot_config.y,
                                                  self.robot_config.theta,
                                                  start_time])
                self.camera_pos_history.append(self.camera_pose[:2, 3].tolist())
                self.camera_pan_history.append(self.pan_angle)
                self.measurements_history.append(measurements.tolist())
                if self.hparams.simulation:
                    self.agents_pos_history.append(agents_pos.tolist())
                end_time = time.time()
                deltat = end_time - start_time
                self.time_history.append([deltat, start_time])

            rate.sleep()

def main():
    rospy.init_node('tiago_camera_detection', log_level=rospy.INFO)
    rospy.loginfo('TIAGo camera detection module [OK]')

    camera_detection_manager = CameraDetectionManager()
    prof_filename = '/tmp/camera_detection.prof'
    cProfile.runctx(
        'camera_detection_manager.run()',
        globals=globals(),
        locals=locals(),
        filename=prof_filename
    )
