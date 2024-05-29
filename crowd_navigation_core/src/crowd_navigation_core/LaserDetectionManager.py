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
from scipy.spatial.transform import Rotation as R

from crowd_navigation_core.Hparams import *
from crowd_navigation_core.utils import *

import gazebo_msgs.msg
import sensor_msgs.msg
import crowd_navigation_msgs.msg

class LaserDetectionManager:
    '''
    Extract the agents' representative 2D-point from the laser sensor 
    '''
    def __init__(self):
        self.data_lock = threading.Lock()

        # Set status, two possibilities:
        # WAITING: waiting for the initial robot configuration
        # READY: reading from the laser
        self.status = Status.WAITING

        self.robot_config = Configuration(0.0, 0.0, 0.0)
        self.hparams = Hparams()
        self.laser_scan_nonrt = None
        if self.hparams.simulation:
            self.agents_pos_nonrt = np.zeros((self.hparams.n_agents, 2))
            self.agents_name = ['actor_{}'.format(i) for i in range(self.hparams.n_agents)]

        # Set variables to store data
        if self.hparams.log:
            self.time_history = []
            self.scans_history = []
            self.measurements_history = []
            self.robot_config_history = []
            self.laser_pos_history = []
            self.boundary_vertexes = []
            if self.hparams.simulation:
                self.agents_pos_history = []

        # Setup reference frames:
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'
        self.laser_frame = 'base_laser_link'

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

        # Setup subscriber to model_states topic
        model_states_topic = "/gazebo/model_states"
        rospy.Subscriber(
            model_states_topic,
            gazebo_msgs.msg.ModelStates,
            self.gazebo_model_states_callback
        )

        # Setup publisher to laser_measurements topic
        measurements_topic = 'laser_measurements'
        self.measurements_publisher = rospy.Publisher(
            measurements_topic,
            crowd_navigation_msgs.msg.MeasurementsSetStamped,
            queue_size=1
        )

    def laser_scan_callback(self, msg):
        self.laser_scan_nonrt = LaserScan.from_message(msg)

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
        
    def get_laser_pose(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.laser_frame, rospy.Time()
            )
            quat = np.array([transform.transform.rotation.x,
                             transform.transform.rotation.y,
                             transform.transform.rotation.z,
                             transform.transform.rotation.w])
            
            pos = np.array([transform.transform.translation.x, 
                            transform.transform.translation.y, 
                            transform.transform.translation.z])
            
            rotation_matrix = R.from_quat(quat).as_matrix()
            self.laser_pose = np.eye(4)
            self.laser_pose[:3, :3] = rotation_matrix
            self.laser_pose[:3, 3] = pos
            return True
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing laser pose")
            return False

    def polar2cartesian(self, polar):
        idx = polar[0]
        distance = polar[1]
        angle = self.laser_scan.angle_min + idx * self.laser_scan.angle_increment
        cartesian = np.array([distance * math.cos(angle),
                              distance * math.sin(angle)])
        return cartesian
    
    def data_preprocessing(self):
        scans = self.laser_scan.ranges
        output_points = []

        # Delete the first and last 'offset' laser scan ranges (wrong measurements?)
        offset = self.hparams.offset
        scans = np.delete(scans, range(offset), 0)
        scans = np.delete(scans, range(len(scans) - offset, len(scans)), 0)

        for idx, value in enumerate(scans):
            if value != np.inf and value >= self.laser_scan.range_min:
                point_las = self.polar2cartesian((idx + offset, value))
                homo_point_las = np.append(point_las, np.append(self.laser_pose[2,3], 1))
                point_wrld = np.matmul(self.laser_pose, homo_point_las)[:2]
                if not is_outside(point_wrld, self.hparams.vertexes, self.hparams.normals):
                    output_points.append(point_wrld.tolist())

        return output_points

    def data_clustering(self, observations):
        robot_position = self.robot_config.get_q()[:2]
        if self.hparams.selection_mode == SelectionMode.CLOSEST:
            window_size = self.hparams.avg_win_size

        if len(observations) != 0:
            k_means = DBSCAN(eps=self.hparams.eps,
                             min_samples=self.hparams.min_samples)
            clusters = k_means.fit_predict(np.array(observations))
            dynamic_n_clusters = max(clusters) + 1
            core_points = np.zeros((dynamic_n_clusters, 2))

            if self.hparams.selection_mode == SelectionMode.CLOSEST:
                clusters_points = [[] for _ in range(dynamic_n_clusters)]

                for id, point in zip(clusters, observations):
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
            elif self.hparams.selection_mode == SelectionMode.AVERAGE:
                n_points = np.zeros((dynamic_n_clusters,))       

                for id, point in zip(clusters, observations):
                    if id != -1:
                        n_points[id] += 1
                        core_points[id] = core_points[id] + (point - core_points[id]) / n_points[id]

            core_points, _ = sort_by_distance(core_points, robot_position)
            core_points = core_points[:self.hparams.n_filters]
            core_points = np.column_stack((core_points, np.full(core_points.shape[0], -1)))
        else:
            core_points = np.array([])

        return core_points

    def log_values(self):
        output_dict = {}
        output_dict['cpu_time'] = self.time_history
        output_dict['laser_scans'] = self.scans_history
        output_dict['measurements'] = self.measurements_history
        output_dict['robot_config'] = self.robot_config_history
        output_dict['laser_position'] = self.laser_pos_history
        output_dict['frequency'] = self.hparams.laser_detector_frequency
        output_dict['b'] = self.hparams.b
        output_dict['n_filters'] = self.hparams.n_filters
        output_dict['n_points'] = self.hparams.n_points
        for i in range(self.hparams.n_points):
            self.boundary_vertexes.append(self.hparams.vertexes[i].tolist())
        output_dict['boundary_vertexes'] = self.boundary_vertexes
        output_dict['base_radius'] = self.hparams.base_radius
        output_dict['simulation'] = self.hparams.simulation
        if self.hparams.simulation:
            output_dict['n_agents'] = self.hparams.n_agents
            output_dict['agents_pos'] = self.agents_pos_history
            output_dict['agent_radius'] = self.hparams.ds_cbf
        output_dict['laser_offset'] = self.hparams.offset
        output_dict['min_samples'] = self.hparams.min_samples
        output_dict['epsilon'] = self.hparams.eps
        output_dict['angle_min'] = self.laser_scan.angle_min
        output_dict['angle_max'] = self.laser_scan.angle_max
        output_dict['angle_inc'] = self.laser_scan.angle_increment
        output_dict['range_min'] = self.laser_scan.range_min
        output_dict['range_max'] = self.laser_scan.range_max
        if self.hparams.selection_mode == SelectionMode.CLOSEST:
            output_dict['win_size'] = self.hparams.avg_win_size

        # log the data in a .json file
        log_dir = self.hparams.log_dir
        filename = self.hparams.laser_detector_file
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, filename)
        with open(log_path, 'w') as file:
            json.dump(output_dict, file)     

    def run(self):
        rate = rospy.Rate(self.hparams.laser_detector_frequency)

        if self.hparams.n_filters == 0:
            rospy.logwarn("No agent considered, laser detection disabled")
            return

        if self.hparams.perception == Perception.FAKE or self.hparams.perception == Perception.CAMERA:
            rospy.logwarn("Laser detection disabled")
            return

        if self.hparams.log:
            rospy.on_shutdown(self.log_values)

        while not rospy.is_shutdown():
            start_time = time.time()

            if self.status == Status.WAITING:
                if self.update_configuration() and self.get_laser_pose():
                    self.status = Status.READY
                else:
                    rate.sleep()
                    continue

            if self.laser_scan_nonrt is None:
                rospy.logwarn("Missing laser data")
                rate.sleep()
                continue

            with self.data_lock:
                self.update_configuration()
                self.get_laser_pose()
                self.laser_scan = self.laser_scan_nonrt
                if self.hparams.simulation:
                    agents_pos = self.agents_pos_nonrt

            # Perform data preprocessing
            observations = self.data_preprocessing()
            # Perform data clustering
            measurements = self.data_clustering(observations)

            # Create measurements message
            measurements_set = MeasurementsSet()
            for measurement in measurements:
                measurements_set.append(Measurement(measurement[0], measurement[1], measurement[2]))
            measurements_set_stamped = MeasurementsSetStamped(rospy.Time.from_sec(start_time),
                                                              'map',
                                                              measurements_set)
            measurements_set_stamped_msg = MeasurementsSetStamped.to_message(measurements_set_stamped)
            self.measurements_publisher.publish(measurements_set_stamped_msg)

            # Update logged data
            if self.hparams.log:
                self.robot_config_history.append([self.robot_config.x,
                                                  self.robot_config.y,
                                                  self.robot_config.theta,
                                                  start_time])
                self.laser_pos_history.append(self.laser_pose[:2, 3].tolist())
                self.scans_history.append(observations)
                self.measurements_history.append(measurements.tolist())
                if self.hparams.simulation:
                    self.agents_pos_history.append(agents_pos.tolist())
                end_time = time.time()
                deltat = end_time - start_time
                self.time_history.append([deltat, start_time])

            rate.sleep()

def main():
    rospy.init_node('tiago_laser_detection', log_level=rospy.INFO)
    rospy.loginfo('TIAGo laser detection module [OK]')

    laser_detection_manager = LaserDetectionManager()
    prof_filename = '/tmp/laser_detection.prof'
    cProfile.runctx(
        'laser_detection_manager.run()',
        globals=globals(),
        locals=locals(),
        filename=prof_filename
    )


            