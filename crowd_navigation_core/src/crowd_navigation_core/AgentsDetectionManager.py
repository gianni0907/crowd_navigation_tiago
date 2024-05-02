import numpy as np
import threading
import rospy
import tf2_ros
import cv2
import os
import cProfile
from cv_bridge import CvBridge
from ultralytics import YOLO

from crowd_navigation_core.Hparams import *
from crowd_navigation_core.utils import *

import sensor_msgs.msg

class AgentsDetectionManager:
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
        self.cv_image = None
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
        self.

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

        # Setup publisher for the processed images
        processed_image_topic = '/image'
        self.processed_image_publisher = rospy.Publisher(
            processed_image_topic,
            sensor_msgs.msg.Image,
            queue_size=1
        )

    def image_callback(self, data):
        try:
            self.data_lock.acquire()
            self.cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            self.data_lock.release()
        except Exception as e:
            rospy.logerr(e)


    def run(self):
        rate = rospy.Rate(self.frequency)

        while not rospy.is_shutdown():
            if self.cv_image is None:
                rospy.logwarn("Missing images")
                rate.sleep()
                continue

            results = self.model.predict(self.cv_image, conf=0.5)
            for result in results:
                image = result.plot()
                self.processed_image_publisher.publish(self.bridge.cv2_to_imgmsg(image))
            rate.sleep()

def main():
    rospy.init_node('tiago_agents_detection', log_level=rospy.INFO)
    rospy.loginfo('TIAGo agents detection module [OK]')

    agents_detection_manager = AgentsDetectionManager()
    prof_filename = '/tmp/agents_detection.prof'
    cProfile.runctx(
        'agents_detection_manager.run()',
        globals=globals(),
        locals=locals(),
        filename=prof_filename
    )
