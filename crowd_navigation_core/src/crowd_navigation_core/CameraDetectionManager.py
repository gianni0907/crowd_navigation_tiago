import numpy as np
import threading
import rospy
import tf2_ros
import cv2
import os
import cProfile
import torch
from cv_bridge import CvBridge
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R
from image_geometry import PinholeCameraModel as PC

from crowd_navigation_core.Hparams import *
from crowd_navigation_core.utils import *

import sensor_msgs.msg

class CameraDetectionManager:
    '''
    Detect surrounding agents from the RGB-D camera images
    '''
    def __init__(self):
        self.data_lock = threading.Lock()

        self.status = Status.WAITING # WAITING for the initial robot state
        self.robot_state = State(0.0, 0.0, 0.0, 0.0, 0.0)
        self.hparams = Hparams()
        self.bridge = CvBridge()
        self.model = YOLO("yolov8n.pt")
        self.n_actors = self.hparams.n_actors
        self.core_points = np.zeros((self.n_actors, 2))

        self.frequency = self.hparams.controller_frequency

        # Set variables to store data
        if self.hparams.log:
            if not self.hparams.fake_sensing:
                self.core_points_history = []
                self.robot_state_history = []

        # Setup reference frames
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'
        self.camera_frame = 'xtion_rgb_optical_frame'

        # Setup TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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

        # Setup publisher for the processed images
        processed_image_topic = '/image'
        self.processed_image_publisher = rospy.Publisher(
            processed_image_topic,
            sensor_msgs.msg.Image,
            queue_size=1
        )

    def image_callback(self, data):
        with self.data_lock:
            try:
                self.cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            except Exception as e:
                rospy.logerr(e)

    def depth_callback(self, data):
        with self.data_lock:
            try:
                self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            except Exception as e:
                rospy.logerr(e)

    def tf2q(self, transform):
        q = transform.transform.rotation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.robot_state.theta = theta
        self.robot_state.x = transform.transform.translation.x + self.hparams.b * math.cos(theta)
        self.robot_state.y = transform.transform.translation.y + self.hparams.b * math.sin(theta)

    def update_state(self):
        try:
            # Update [x, y, theta]
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_footprint_frame, rospy.Time()
            )
            self.tf2q(transform)
            # Update [v, omega]
            self.robot_state.v = self.hparams.wheel_radius * 0.5 * \
                (self.wheels_vel[self.hparams.r_wheel_idx] + self.wheels_vel[self.hparams.l_wheel_idx])
            
            self.robot_state.omega = (self.hparams.wheel_radius / self.hparams.wheel_separation) * \
                (self.wheels_vel[self.hparams.r_wheel_idx] - self.wheels_vel[self.hparams.l_wheel_idx])
            return True
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing current state")
            return False

    def get_camera_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.camera_frame, rospy.Time()
            )
            quat = np.array([transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z,
                             transform.transform.rotation.w])
            
            pos = np.array([transform.transform.translation.x, 
                            transform.transform.translation.y, 
                            transform.transform.translation.z])
            
            rotation_matrix = R.from_quat(quat).as_matrix()
            self.camera_pose = np.eye(4)
            self.camera_pose[:3, :3] = rotation_matrix
            self.camera_pose[:3, 3] = pos
            return True
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing camera pose")
            return False

    def run(self):
        cam_info = rospy.wait_for_message("/xtion/rgb/camera_info", sensor_msgs.msg.CameraInfo, timeout=None)
        imgproc = PC()
        imgproc.fromCameraInfo(cam_info)

        rate = rospy.Rate(self.frequency)
        while not rospy.is_shutdown():
            if self.cv_image is None:
                rospy.logwarn("Missing images")
                rate.sleep()
                continue

            self.get_camera_pose()
            results = self.model(self.cv_image, conf=0.5)
            for result in results:
                labels, cords = result.boxes.cls, result.boxes.xyxyn
                for label, cord in zip(labels, cords):
                    if label == 0:
                        box_center = torch.tensor([(cord[0] + cord[2]) / 2 * self.cv_image.shape[1],
                                                   (cord[1] + cord[3]) / 2 * self.cv_image.shape[0]]).to("cpu")
                        x_c = int(torch.round(box_center[0]).item())
                        y_c = int(torch.round(box_center[1]).item())
                        # center = (x_c, y_c)
                        # cv2.circle(self.cv_image, center, radius=5, color=(0, 0, 255), thickness=-1)
                        depth = self.depth_image[y_c, x_c]
                        obj_coord = np.array(imgproc.projectPixelTo3dRay(box_center))
                        scale = depth / obj_coord[2]
                        obj_coord = obj_coord * scale
                        homo_obj_coord = np.array([obj_coord[0],
                                                   obj_coord[1],
                                                   obj_coord[2],
                                                   1])
                        wrld_obj_coord = np.matmul(self.camera_pose, homo_obj_coord)
                image = result.plot()
                self.processed_image_publisher.publish(self.bridge.cv2_to_imgmsg(image)) 
            rate.sleep()

def main():
    rospy.init_node('tiago_camera_detection', log_level=rospy.INFO)
    rospy.loginfo('TIAGo camera detection module [OK]')

    agents_detection_manager = CameraDetectionManager()
    prof_filename = '/tmp/agents_detection.prof'
    cProfile.runctx(
        'agents_detection_manager.run()',
        globals=globals(),
        locals=locals(),
        filename=prof_filename
    )
