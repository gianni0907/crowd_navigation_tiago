import numpy as np
import os
import json
import math
import rospy
import threading
import sensor_msgs.msg
import tf2_ros
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

from my_tiago_controller.utils import *
from my_tiago_controller.Hparams import *
from my_tiago_controller.Status import *
from my_tiago_controller.FSM import *

import my_tiago_msgs.srv
import my_tiago_msgs.msg

def data_clustering(scans, tiago_state, angle_min, angle_incr):
    scans_xy_abs = []
    scans_polar = []
    for element in scans:
        if element[1] != np.inf:
            scans_polar.append(element)
            scans_xy_abs.append(polar2absolute(element, tiago_state, angle_min, angle_incr))

    if len(scans_xy_abs) != 0:
        k_means = DBSCAN(eps=1, min_samples=2)
        clusters = k_means.fit_predict(np.array(scans_xy_abs))
        dynamic_n_clusters = max(clusters) + 1
        if(min(clusters) == -1):
            print("Noisy samples")
        
        polar_core_points = np.zeros((dynamic_n_clusters, 2))

        for cluster_i in range(dynamic_n_clusters):
            min_distance = np.inf
            for id, (idx_scan, polar_scan) in zip(clusters, enumerate(scans_polar)):
                if id == cluster_i:
                    if polar_scan[1] < min_distance:
                        min_distance = polar_scan[1]
                        polar_core_points[cluster_i] = scans_polar[idx_scan]

        polar_core_points = polar_core_points.tolist()        
        polar_core_points.sort(key = lambda x: x[1], reverse = False)
        polar_core_points = np.array(polar_core_points[0:Hparams.n_clusters])
    else:
        polar_core_points = np.array([])

    return polar_core_points

class CrowdPredictionManager:
    '''
    From the laser scans input predict the motion of the actors
    '''
    def __init__(self):
        self.data_lock = threading.Lock()

        # Set status
        # 2 possibilities:
        #   WAITING for the initial robot state
        #   READY to perceive and act
        self.status = Status.WAITING # WAITING for the initial robot state

        self.robot_state = State(0.0, 0.0, 0.0, 0.0, 0.0)
        self.hparams = Hparams()
        self.laser_scan = None
        self.n_actors = self.hparams.n_actors
        self.n_clusters = self.hparams.n_clusters

        self.kalman_infos = {}
        kalman_names = ['KF_{}'.format(i + 1) for i in range(self.n_actors)]
        self.kalman_infos = {key: list() for key in kalman_names}

        self.N_horizon = self.hparams.N_horizon
        self.frequency = self.hparams.controller_frequency

        # Setup reference frames:
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'

        # Setup TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Setup subscriber for joint_states topic
        state_topic = '/joint_states'
        rospy.Subscriber(
            state_topic,
            sensor_msgs.msg.JointState,
            self.joint_states_callback
        )

        # Setup subscriber to scan_raw topic
        scan_topic = '/scan_raw'
        rospy.Subscriber(
            scan_topic,
            sensor_msgs.msg.LaserScan,
            self.laser_scan_callback
        )

        # Setup publisher for crowd motion prediction:
        crowd_prediction_topic = 'crowd_motion_prediction'
        self.crowd_motion_prediction_publisher = rospy.Publisher(
            crowd_prediction_topic,
            my_tiago_msgs.msg.CrowdMotionPredictionStamped,
            queue_size=1
        )

    def joint_states_callback(self, msg):
        self.wheels_vel = np.array([msg.velocity[13], msg.velocity[12]])

    def laser_scan_callback(self, msg):
        self.data_lock.acquire()
        self.laser_scan = LaserScan.from_message(msg)
        self.data_lock.release()

    def set_from_tf_transform(self, transform):
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
            self.set_from_tf_transform(transform)
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

    def propagate_state(self, state, N):
        predictions = [self.hparams.nullstate for _ in range(N)]
        time = 0
        dt = 1 / self.frequency
        for i in range(N):
            time = dt * (i)
            predictions[i] = self.predict_next_state(state, time)

        return predictions

    def predict_next_state(self, state, dt):
        F = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        next_state = np.matmul(F, state)

        return next_state

    def log_values(self):
        output_dict = {}
        output_dict['kfs'] = self.kalman_infos
        
        # log the data in a .json file
        log_dir = '/tmp/crowd_navigation_tiago/data'
        filename = self.hparams.prediction_file
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, filename)
        with open(log_path, 'w') as file:
            json.dump(output_dict, file)


    def run(self):
        rate = rospy.Rate(self.frequency)
        if self.hparams.log:
            rospy.on_shutdown(self.log_values)

        if self.n_actors == 0:
            rospy.logwarn("No actors available")
            return
        
        fsms = [FSM(self.hparams) for _ in range(self.n_clusters)]

        while not rospy.is_shutdown():
            time = rospy.get_time()
            
            if self.status == Status.WAITING:
                if self.update_state():
                    self.status = Status.READY
                    print("Initial state ****************************")
                    print(self.robot_state)
                    print("******************************************")
                else:
                    rate.sleep()
                    continue
            
            self.data_lock.acquire()
            self.update_state()
            self.data_lock.release()
            
            if self.laser_scan is None:
                rospy.logwarn("Missing laser scan info")
                rate.sleep()
                continue

            # Reset crowd_motion_prediction message
            crowd_motion_prediction = CrowdMotionPrediction()
            
            angle_min = self.laser_scan.angle_min
            angle_max = self.laser_scan.angle_max
            angle_increment = self.laser_scan.angle_increment
            offset = 20

            scanlist = []
            for idx, value in enumerate(self.laser_scan.ranges):
                scanlist.append((idx, value))

            # Delete the first and last 20 laser scan ranges (wrong measurements?)
            scanlist = np.delete(scanlist, range(offset), 0)
            scanlist = np.delete(scanlist, range(len(scanlist) - offset, len(scanlist)), 0)

            # Perform data clustering
            actors_polar_position = data_clustering(scanlist,
                                                    self.robot_state,
                                                    angle_min,
                                                    angle_increment)
            actors_absolute_position = np.copy(actors_polar_position)
            for (i, dist) in enumerate(actors_absolute_position):
                actors_absolute_position[i] = polar2absolute(actors_absolute_position[i],
                                                             self.robot_state,
                                                             angle_min,
                                                             angle_increment)

            # Perform data association
            # predict next positions according to fsms
            predicted_positions = np.zeros((self.hparams.n_clusters, 2))
            associated_clusters = []

            for (i, fsm) in enumerate(fsms):
                if fsm.next_state in (FSMStates.ACTIVE, FSMStates.HOLD):
                    predicted_state = self.propagate_state(fsm.current_estimate, 1)[0]
                elif fsm.next_state is FSMStates.START:
                    predicted_state = fsm.current_estimate
                else:
                    predicted_state = self.hparams.nullstate
                predicted_positions[i] = predicted_state[:2]
            
            # compute pairwise distance between predicted positions and positions from clusters
            if actors_absolute_position.shape[0] > 0:
                distances = cdist(predicted_positions, actors_absolute_position)
            else:
                distances = None

            # match each fsm to the closest cluster centroid
            for (i, fsm) in enumerate(fsms):
                fsm_state = fsm.next_state
                
                if self.hparams.log:
                    self.kalman_infos['KF_{}'.format(i + 1)].append([FSMStates.print(fsm.state),
                                                                     FSMStates.print(fsm_state),
                                                                     time])

                # find the closest available cluster to the fsm's prediction
                min_dist = np.inf
                cluster = None
                if distances is not None:
                    for j in range(distances.shape[1]):
                        if j in associated_clusters:
                            continue
                        distance = distances[i, j]
                        if distance < min_dist:
                            min_dist = distance
                            cluster = j
                if cluster is not None:
                    measure = actors_absolute_position[cluster]
                    fsm.update(time, measure)
                    associated_clusters.append(cluster)
                else:
                    measure = np.array([0.0, 0.0])
                    fsm.update(time, measure)

                current_estimate = fsm.current_estimate
                predictions = self.propagate_state(current_estimate, self.N_horizon)
                predicted_positions = [Position(0.0, 0.0) for _ in range(self.N_horizon)]
                predicted_velocities = [Velocity(0.0, 0.0) for _ in range(self.N_horizon)]
                for j in range(self.N_horizon):
                    predicted_positions[j] = Position(predictions[j][0], predictions[j][1])
                    predicted_velocities[j] = Velocity(predictions[j][2], predictions[j][3])
                
                crowd_motion_prediction.append(
                        MotionPrediction(predicted_positions, predicted_velocities)
                )

            crowd_motion_prediction_stamped = CrowdMotionPredictionStamped(rospy.Time.from_sec(time),
                                                                            'map',
                                                                            crowd_motion_prediction)
            crowd_motion_prediction_stamped_msg = CrowdMotionPredictionStamped.to_message(crowd_motion_prediction_stamped)
            self.crowd_motion_prediction_publisher.publish(crowd_motion_prediction_stamped_msg)

            rate.sleep()

def main():
    rospy.init_node('tiago_crowd_prediction', log_level=rospy.INFO)
    rospy.loginfo('TIAGo crowd prediction module [OK]')

    crowd_prediction_manager = CrowdPredictionManager()
    crowd_prediction_manager.run()