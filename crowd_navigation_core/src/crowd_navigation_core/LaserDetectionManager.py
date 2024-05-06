import numpy as np
import math
import time
import os
import json
import threading
import cProfile
import tf2_ros
import rospy
from sklearn.cluster import DBSCAN

from crowd_navigation_core.Hparams import *
from crowd_navigation_core.utils import *

import sensor_msgs.msg
import crowd_navigation_msgs.msg

def polar2relative(scan, angle_min, angle_incr):
    relative_laser_pos = Hparams.relative_laser_pos
    idx = scan[0]
    distance = scan[1]
    angle = angle_min + idx * angle_incr
    xy_relative = np.array([distance * math.cos(angle) + relative_laser_pos[0],
                            distance * math.sin(angle) + relative_laser_pos[1]])
    return xy_relative

def polar2absolute(scan, config, angle_min, angle_incr):
    xy_relative = polar2relative(scan, angle_min, angle_incr)
    xy_absolute = z_rotation(config.theta, xy_relative) + np.array([config.x, config.y])
    return xy_absolute

def moving_average(points, window_size=1):
    smoothed_points = np.zeros(points.shape)
    
    for i in range(points.shape[0]):
        # Compute indices for the moving window
        start_idx = np.max([0, i - window_size // 2])
        end_idx = np.min([points.shape[0], i + window_size // 2 + 1])
        smoothed_points[i] = np.sum(points[start_idx : end_idx], 0) / (end_idx - start_idx)

    return smoothed_points

def data_preprocessing(scans, config, range_min, angle_min, angle_incr):
    n_edges = Hparams.n_points
    vertexes = Hparams.vertexes
    normals = Hparams.normals
    absolute_scans = []

    # Delete the first and last 'offset' laser scan ranges (wrong measurements?)
    offset = Hparams.offset
    scans = np.delete(scans, range(offset), 0)
    scans = np.delete(scans, range(len(scans) - offset, len(scans)), 0)

    for idx, value in enumerate(scans):
        outside = False
        if value != np.inf and value >= range_min:
            absolute_scan = polar2absolute((idx + offset, value), config, angle_min, angle_incr)
            for i in range(n_edges):
                vertex = vertexes[i]
                if np.dot(normals[i], absolute_scan - vertex) < 0.0:
                    outside = True
                    break
            if not outside:
                absolute_scans.append(absolute_scan.tolist())

    return absolute_scans

def data_clustering(absolute_scans, config):
    robot_position = config.get_q()[:2]
    selection_mode = Hparams.selection_mode
    eps = Hparams.eps
    min_samples = Hparams.min_samples
    if selection_mode == SelectionMode.CLOSEST:
        window_size = Hparams.avg_win_size

    if len(absolute_scans) != 0:
        k_means = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = k_means.fit_predict(np.array(absolute_scans))
        dynamic_n_clusters = max(clusters) + 1
        core_points = np.zeros((dynamic_n_clusters, 2))

        if selection_mode == SelectionMode.CLOSEST:
            clusters_points = [[] for _ in range(dynamic_n_clusters)]

            for id, point in zip(clusters, absolute_scans):
                if id != -1:
                    clusters_points[id].append(point)

            for i in range(dynamic_n_clusters):
                cluster_points = np.array(clusters_points[i])
                smoothed_points = moving_average(cluster_points, window_size)
                min_distance = np.inf
                for point in (smoothed_points):
                    distance = euclidean_distance(point, robot_position)          
                    if distance < min_distance:
                        min_distance = distance
                        core_points[i] = point
        elif selection_mode == SelectionMode.AVERAGE:
            n_points = np.zeros((dynamic_n_clusters,))       

            for id, point in zip(clusters, absolute_scans):
                if id != -1:
                    n_points[id] += 1
                    core_points[id] = core_points[id] + (point - core_points[id]) / n_points[id]

        core_points, _ = sort_by_distance(core_points, robot_position)
        core_points = core_points[:Hparams.n_clusters]
    else:
        core_points = np.array([])

    return core_points

class LaserDetectionManager:
    '''
    Extract the agents' representative 2D-point from the laser sensor 
    '''
    def __init__(self):
        self.data_lock = threading.Lock()

        # Set status, two possibilities:
        # WAITING: waiting for the initial robot configuration
        # READY: reading from the laser and moving
        self.status = Status.WAITING

        self.robot_config = Configuration(0.0, 0.0, 0.0)
        self.laser_scan = None
        self.hparams = Hparams()
        self.measurements = np.zeros((self.hparams.n_clusters, 2))

        # Set variables to store data
        if self.hparams.log:
            self.time_history = []
            self.scans_history = []

        # Setup reference frames:
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'

        # Setup TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Setup subscriber to scan_raw topic
        scan_topic = '/scan_raw'
        rospy.Subscriber(
            scan_topic,
            sensor_msgs.msg.LaserScan,
            self.laser_scan_callback
        )

        # Setup publisher to laser_measurements topic
        measurements_topic = 'laser_measurements'
        self.measurements_publisher = rospy.Publisher(
            measurements_topic,
            crowd_navigation_msgs.msg.MeasurementsStamped,
            queue_size=1
        )

    def laser_scan_callback(self, msg):
        self.data_lock.acquire()
        self.laser_scan = LaserScan.from_message(msg)
        self.data_lock.release()

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
        
    def find_core_points(self):
        # Perform data preprocessing
        self.absolute_scans = []
        self.absolute_scans = data_preprocessing(self.laser_scan.ranges,
                                                 self.robot_config,
                                                 self.laser_scan.range_min,
                                                 self.laser_scan.angle_min,
                                                 self.laser_scan.angle_increment)

        # Perform data clustering
        core_points = data_clustering(self.absolute_scans,
                                      self.robot_config)
            
        with self.data_lock:
            self.measurements = core_points

    def log_values(self):
        output_dict = {}
        output_dict['cpu_time'] = self.time_history
        output_dict['laser_scans'] = self.scans_history
        output_dict['laser_offset'] = self.hparams.offset
        output_dict['laser_relative_pos'] = self.hparams.relative_laser_pos.tolist()
        output_dict['min_samples'] = self.hparams.min_samples
        output_dict['epsilon'] = self.hparams.eps
        if self.hparams.selection_mode == SelectionMode.CLOSEST:
            output_dict['win_size'] = self.hparams.avg_win_size
        output_dict['angle_min'] = self.laser_scan.angle_min
        output_dict['angle_max'] = self.laser_scan.angle_max
        output_dict['angle_inc'] = self.laser_scan.angle_increment
        output_dict['range_min'] = self.laser_scan.range_min
        output_dict['range_max'] = self.laser_scan.range_max

        # log the data in a .json file
        log_dir = '/tmp/crowd_navigation_tiago/data'
        filename = self.hparams.predictor_file
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, filename)
        with open(log_path, 'w') as file:
            json.dump(output_dict, file)     

    def run(self):
        rate = rospy.Rate(self.hparams.controller_frequency)

        while not rospy.is_shutdown():
            start_time = time.time()

            if self.status == Status.WAITING:
                with self.data_lock:
                    if self.update_configuration():
                        self.status = Status.READY
                    else:
                        rate.sleep()
                        continue

            if self.laser_scan is None:
                rospy.logwarn("Missing laser info")
                rate.sleep()
                continue

            with self.data_lock:
                self.update_configuration()

            self.find_core_points()

            # Update core_points message
            measurements = Measurements(self.core_points)
            core_points_stamped = CorePointsStamped(rospy.Time.from_sec(start_time),
                                                    'map',
                                                    core_points)
            core_points_stamped_msg = CorePointsStamped.to_message(core_points_stamped)
            self.core_points_publisher.publish(core_points_stamped_msg)

            if self.hparams.log:
                self.scans_history.append(self.absolute_scans)
                end_time = time.time()
                deltat = end_time - start_time
                self.time_history.append([deltat, start_time])

            rate.sleep()

def main():
    rospy.init_node('tiago_laser_detection', log_level=rospy.INFO)
    rospy.loginfo('TIAGo laser detection module [OK]')

    laser_manager = LaserDetectionManager()
    prof_filename = '/tmp/laser.prof'
    cProfile.runctx(
        'laser_manager.run()',
        globals=globals(),
        locals=locals(),
        filename=prof_filename
    )


            