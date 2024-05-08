import numpy as np
import os
import time
import json
import math
import rospy
import threading
import tf2_ros
import cProfile
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

from crowd_navigation_core.utils import *
from crowd_navigation_core.Hparams import *
from crowd_navigation_core.FSM import *

import sensor_msgs.msg
import crowd_navigation_msgs.srv
import crowd_navigation_msgs.msg

def data_association(predictions, covariances, measurements):
    n_measurements = measurements.shape[0]
    n_fsms = predictions.shape[0]

    # Heuristics to consider for the associations: gating, best friend, lonely best friend
    # heuristics param
    gating_tau = 10 # maximum Mahalanobis distance threshold
    gamma_threshold = 1e-1 # lonely best friends threshold
    # consider 2 arrays of dimension [n_measurements] containing the following association info:
    #   fsm indices
    #   association value (euclidean distance)
    fsm_indices = -1 * np.ones(n_measurements, dtype=int)
    distances = np.ones(n_measurements) * np.inf

    if n_fsms == 0 or n_measurements == 0:
        return fsm_indices

    A_mat = np.zeros((n_measurements, n_fsms)) # [n_measurements x n_fsms] Association matrix
    for i in range(n_fsms):
        info_mat = np.linalg.inv(covariances[i])
        for j in range(n_measurements):
            diff = measurements[j] - predictions[i]
            A_mat[j, i] = np.sqrt(diff @ info_mat @ diff.T)

    for j in range(n_measurements):
        # compute row minimum
        d_ji = np.min(A_mat[j, :])
        min_idx = np.argmin(A_mat[j, :])

        # gating
        if (d_ji < gating_tau):
            fsm_indices[j] = min_idx
            distances[j] = d_ji
        else:
            fsm_indices[j] = -1
            distances[j] = d_ji

    # best friends
    for j in range(n_measurements):    
        proposed_est = fsm_indices[j]
        d_ji = distances[j]
        if proposed_est != -1:
            # compute column minimum
            col_min = np.min(A_mat[:, proposed_est])

            if d_ji != col_min:
                fsm_indices[j] = -1


    # lonely best friends
    if n_fsms > 1 and n_measurements > 1:
        for j in range(n_measurements):
            proposed_est = fsm_indices[j]
            d_ji = distances[j]

            if proposed_est == -1:
                continue

            # take second best value of the row
            ordered_row = np.sort(A_mat[j, :])
            second_min_row = ordered_row[1]

            # take second best value of the col
            ordered_col = np.sort(A_mat[:, proposed_est])
            second_min_col = ordered_col[1]

            # check association ambiguity
            if (second_min_row - d_ji) < gamma_threshold or (second_min_col - d_ji) < gamma_threshold:
                fsm_indices[j] = -1
    
    return fsm_indices
class CrowdPredictionManager:
    '''
    Given a set of measurements, i.e., agents' representative 2D-points,
    predict their current state (position and velocity) and future motion
    '''
    def __init__(self):
        self.data_lock = threading.Lock()

        # Set status
        # 2 scenarios:
        #   self.hparams.fake_sensing == True -> 3 possible status
        #       WAITING for the initial robot state
        #       READY to get actors trajectory
        #       MOVING when actors are moving
        #   self.hparams.fake_sensing == False -> 2 possible status
        #       WAITING for the initial robot state
        #       READY to read from laser scan topic and to move
        self.status = Status.WAITING # WAITING for the initial robot state

        self.sensing = False

        self.robot_state = State(0.0, 0.0, 0.0, 0.0, 0.0)
        self.wheels_vel = np.zeros(2) # [w_r, w_l]
        self.hparams = Hparams()
        if self.hparams.use_kalman:
            self.estimated_actors_state = np.zeros((self.hparams.n_clusters, 4))

        self.measurements_stamped = MeasurementsStamped(rospy.Time.now(),
                                                        'map',
                                                        Measurements())

        # Set variables to store data  
        if self.hparams.log:
            self.kalman_infos = {}
            kalman_names = ['KF_{}'.format(i + 1) for i in range(self.hparams.n_clusters)]
            self.kalman_infos = {key: list() for key in kalman_names}
            self.associations = []
            self.time_history = []
            self.agents_prediction_history = []
            if self.hparams.use_kalman:
                self.fsm_estimates_history = []

        # Setup reference frames:
        self.map_frame = 'map'
        self.base_footprint_frame = 'base_footprint'

        # Setup TF listener:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Setup subscriber to joint_states topic
        state_topic = '/joint_states'
        rospy.Subscriber(
            state_topic,
            sensor_msgs.msg.JointState,
            self.joint_states_callback
        )

        # Setup subscriber to laser_measurements topic
        laser_meas_topic = 'laser_measurements'
        rospy.Subscriber(
            laser_meas_topic,
            crowd_navigation_msgs.msg.MeasurementsStamped,
            self.measurements_stamped_callback
        )

        # Setup publisher to crowd_motion_prediction topic
        crowd_prediction_topic = 'crowd_motion_prediction'
        self.crowd_motion_prediction_publisher = rospy.Publisher(
            crowd_prediction_topic,
            crowd_navigation_msgs.msg.CrowdMotionPredictionStamped,
            queue_size=1
        )

        # Setup ROS Service to set actors trajectories:
        self.set_actors_trajectory_srv = rospy.Service(
            'SetActorsTrajectory',
            crowd_navigation_msgs.srv.SetActorsTrajectory,
            self.set_actors_trajectory_request
        )

    def joint_states_callback(self, msg):
        self.wheels_vel = np.array([msg.velocity[13], msg.velocity[12]])

    def measurements_stamped_callback(self, msg):
        measurements_stamped = MeasurementsStamped.from_message(msg)
        with self.data_lock:
            self.measurements_stamped = measurements_stamped
        self.sensing = True

    def tf2q(self, transform):
        q = transform.transform.rotation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.robot_state.theta = theta
        self.robot_state.x = transform.transform.translation.x + self.hparams.b * math.cos(theta)
        self.robot_state.y = transform.transform.translation.y + self.hparams.b * math.sin(theta)

    def set_actors_trajectory_request(self, request):
        if not self.hparams.fake_sensing:
            rospy.loginfo("Cannot set synthetic trajectories, real sensing is active")
            return crowd_navigation_msgs.srv.SetActorsTrajectoryResponse(False) 
        else:
            if self.status == Status.WAITING:
                rospy.loginfo("Cannot set actors trajectory, robot is not READY")
                return crowd_navigation_msgs.srv.SetActorsTrajectoryResponse(False)            
            elif self.status == Status.READY:
                self.trajectories = CrowdMotionPrediction.from_message(request.trajectories)
                self.status = Status.MOVING
                self.current = 0
                self.trajectory_length = len(self.trajectories.motion_predictions[0].positions)
                rospy.loginfo("Actors trajectory successfully set")
                return crowd_navigation_msgs.srv.SetActorsTrajectoryResponse(True)
            else:
                rospy.loginfo("Cannot set actors trajectory, actors are already moving")
                return crowd_navigation_msgs.srv.SetActorsTrajectoryResponse(False)

    def update_measurements(self):
        if self.hparams.fake_sensing:
            measurements = np.zeros((self.hparams.n_clusters, 2))
            for i in range(self.hparams.n_clusters):
                        measurements[i] = np.array([self.trajectories.motion_predictions[i].positions[self.current].x,
                                                    self.trajectories.motion_predictions[i].positions[self.current].y])
            self.current += 1
            if self.current == self.trajectory_length:
                self.current = 0
        else:
            with self.data_lock:
                measurements_rt = self.measurements_stamped.measurements
            measurements = np.zeros((measurements_rt.size, 2))
            for i in range(measurements_rt.size):
                measurements[i] = np.array([measurements_rt.positions[i].x,
                                            measurements_rt.positions[i].y])
                    
        self.measurements = measurements

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

    def propagate_state(self, state, N):
        predictions = [self.hparams.nullstate for _ in range(N)]
        time = 0
        dt = self.hparams.dt
        for i in range(N):
            time = dt * (i + 1)
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
        output_dict['associations'] = self.associations
        output_dict['cpu_time'] = self.time_history
        output_dict['agents_predictions'] = self.agents_prediction_history
        if self.hparams.use_kalman:
            output_dict['fsm_estimates'] = self.fsm_estimates_history
        
        # log the data in a .json file
        log_dir = self.hparams.log_dir
        filename = self.hparams.predictor_file
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, filename)
        with open(log_path, 'w') as file:
            json.dump(output_dict, file)

    def run(self):
        rate = rospy.Rate(self.hparams.controller_frequency)

        if self.hparams.n_actors == 0:
            rospy.logwarn("No actors available")
            return
        
        if self.hparams.log:
            rospy.on_shutdown(self.log_values)
        
        if self.hparams.use_kalman:
            fsms = [FSM(self.hparams) for _ in range(self.hparams.n_clusters)]

        while not rospy.is_shutdown():
            start_time = time.time()
            
            if self.status == Status.WAITING:
                if self.update_state():
                    self.status = Status.READY
                    print("Initial state ****************************")
                    print(self.robot_state)
                    print("******************************************")
                else:
                    rate.sleep()
                    continue
            
            if self.status == Status.READY and self.hparams.fake_sensing:
                rospy.logwarn("Missing fake sensing data")
                rate.sleep()
                continue
            
            if not self.sensing and not self.hparams.fake_sensing:
                rospy.logwarn("Missing laser scan data")
                rate.sleep()
                continue

            self.data_lock.acquire()
            self.update_state()
            self.data_lock.release()

            # Reset crowd_motion_prediction message
            crowd_motion_prediction = CrowdMotionPrediction()
            
            # Update the actors position (based on fake or real sensing)
            N = self.hparams.N_horizon
            if self.hparams.fake_sensing:
                for i in range(self.hparams.n_clusters):
                    positions = [Position(0.0, 0.0) for _ in range(N)]
                    # Create the prediction within the horizon
                    if self.current + N <= self.trajectory_length:
                        positions = self.trajectories.motion_predictions[i].positions[self.current : self.current + N]
                    elif self.current < self.trajectory_length:
                        positions[:self.trajectory_length - self.current] = \
                            self.trajectories.motion_predictions[i].positions[self.current : self.trajectory_length]
                        for j in range(N - self.trajectory_length + self.current):
                            positions[self.trajectory_length - self.current + j] = \
                                self.trajectories.motion_predictions[i].positions[-1]
                    else:
                        for j in range(N):
                            positions[j] = self.trajectories.motion_predictions[i].positions[-1]

                    crowd_motion_prediction.append(MotionPrediction(positions))
                    
                self.current += 1
                if self.current == self.trajectory_length:
                    self.current = 0

            else:
                self.update_measurements()

                if self.hparams.use_kalman:
                    # Perform data association
                    # predict next positions according to fsms
                    next_predicted_positions = np.zeros((self.hparams.n_clusters, 2))
                    next_positions_cov = np.zeros((self.hparams.n_clusters, 2, 2))

                    for (i, fsm) in enumerate(fsms):
                        fsm_next_state = fsm.next_state
                        if self.hparams.log:
                            self.kalman_infos['KF_{}'.format(i + 1)].append([FSMStates.print(fsm.state),
                                                                            FSMStates.print(fsm_next_state),
                                                                            start_time])
                        if fsm_next_state in (FSMStates.ACTIVE, FSMStates.HOLD):
                            predicted_state = self.propagate_state(fsm.current_estimate, 1)[0]
                            predicted_cov = fsm.kalman_f.Pk
                        elif fsm_next_state is FSMStates.START:
                            predicted_state = fsm.current_estimate
                            predicted_cov = np.eye(4) * 1e-2
                        else:
                            predicted_state = self.hparams.nullstate
                            predicted_cov = np.eye(4) * 1e-2

                        next_predicted_positions[i] = predicted_state[:2]
                        next_positions_cov[i] = predicted_cov[:2, :2]

                    fsm_indices = data_association(next_predicted_positions, next_positions_cov, self.measurements)
                    if self.hparams.log:
                        self.associations.append([fsm_indices.tolist(), start_time])

                    # update each fsm based on the associations
                    for (i, fsm) in enumerate(fsms):
                        associated = False
                        for (j,fsm_idx) in enumerate(fsm_indices):
                            if fsm_idx == i:
                                measure = self.measurements[j]
                                fsm.update(start_time, measure)
                                associated = True
                                break
                        if associated == False:
                            for (k, fsm_idx) in enumerate(fsm_indices):
                                if fsm_idx == -1:
                                    measure = self.measurements[k]
                                    fsm.update(start_time, measure)
                                    fsm_indices[k] = i
                                    associated = True
                                    break
                        if associated == False:
                            measure = None
                            fsm.update(start_time, measure)

                        current_estimate = fsm.current_estimate
                        self.estimated_actors_state[i] = current_estimate
                        predictions = self.propagate_state(current_estimate, N)
                        predicted_positions = [Position(0.0, 0.0) for _ in range(N)]
                        for j in range(N):
                            predicted_positions[j] = Position(predictions[j][0], predictions[j][1])
                        
                        crowd_motion_prediction.append(
                                MotionPrediction(predicted_positions)
                        )
                else:
                    for i in range(self.hparams.n_clusters):
                        if i < self.measurements.shape[0]:
                            current_estimate = np.array([self.measurements[i, 0],
                                                         self.measurements[i, 1],
                                                         0.0,
                                                         0.0])
                        else:
                            current_estimate = self.hparams.nullstate

                        predictions = self.propagate_state(current_estimate, N)
                        predicted_positions = [Position(0.0, 0.0) for _ in range(N)]
                        for j in range(N):
                            predicted_positions[j] = Position(predictions[j][0], predictions[j][1])
                    
                        crowd_motion_prediction.append(
                            MotionPrediction(predicted_positions)
                        )

            crowd_motion_prediction_stamped = CrowdMotionPredictionStamped(rospy.Time.from_sec(start_time),
                                                                           'map',
                                                                           crowd_motion_prediction)
            crowd_motion_prediction_stamped_msg = CrowdMotionPredictionStamped.to_message(crowd_motion_prediction_stamped)
            self.crowd_motion_prediction_publisher.publish(crowd_motion_prediction_stamped_msg)
            
            # Saving data for plots
            if self.hparams.log:
                predicted_trajectory = np.zeros((self.hparams.n_clusters, 2, self.hparams.N_horizon))
                for i in range(self.hparams.n_clusters):
                    motion_prediction = crowd_motion_prediction.motion_predictions[i]
                    for j in range(self.hparams.N_horizon):
                        predicted_trajectory[i, 0, j] = motion_prediction.positions[j].x
                        predicted_trajectory[i, 1, j] = motion_prediction.positions[j].y
                self.agents_prediction_history.append(predicted_trajectory.tolist())

                if self.hparams.use_kalman:
                    self.fsm_estimates_history.append(self.estimated_actors_state.tolist())

                end_time = time.time()
                deltat = end_time - start_time
                self.time_history.append([deltat, start_time])

            rate.sleep()

def main():
    rospy.init_node('tiago_crowd_prediction', log_level=rospy.INFO)
    rospy.loginfo('TIAGo crowd prediction module [OK]')

    crowd_prediction_manager = CrowdPredictionManager()
    prof_filename = '/tmp/crowd_prediction.prof'
    cProfile.runctx(
        'crowd_prediction_manager.run()',
        globals=globals(),
        locals=locals(),
        filename=prof_filename
    )