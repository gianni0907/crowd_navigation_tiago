import numpy as np
import time
import os
import json
import threading
import cProfile
import rospy
from scipy.spatial.distance import cdist

from crowd_navigation_core.utils import *
from crowd_navigation_core.Hparams import *
from crowd_navigation_core.FSM import *

import gazebo_msgs.msg
import crowd_navigation_msgs.srv
import crowd_navigation_msgs.msg

class CrowdPredictionManager:
    '''
    Given a set of measurements, i.e., agents' representative 2D-points,
    predict their current state (position and velocity) and future motion
    '''
    def __init__(self):
        self.data_lock = threading.Lock()

        self.hparams = Hparams()

        # Set variables to store data  
        if self.hparams.log:
            self.time_history = []
            self.kalman_infos = {}
            kalman_names = ['KF_{}'.format(i + 1) for i in range(self.hparams.n_filters)]
            self.kalman_infos = {key: list() for key in kalman_names}
            if self.hparams.use_kalman:
                self.associations = []
                self.estimation_history = []

        if self.hparams.perception == Perception.GTRUTH and self.hparams.simulation:
            self.agents_pos_nonrt = None
            self.agents_name = ['actor_{}'.format(i) for i in range(self.hparams.n_agents)]
            # Setup subscriber to model_states topic
            model_states_topic = "/gazebo/model_states"
            rospy.Subscriber(
                model_states_topic,
                gazebo_msgs.msg.ModelStates,
                self.gazebo_model_states_callback
            )
        else:
            if self.hparams.perception in (Perception.LASER, Perception.BOTH):  
                self.laser_measurements_stamped_nonrt = None
                # Setup subscriber to laser_measurements topic
                laser_meas_topic = 'laser_measurements'
                rospy.Subscriber(
                    laser_meas_topic,
                    crowd_navigation_msgs.msg.MeasurementsSetStamped,
                    self.laser_measurements_callback
                )
            if self.hparams.perception in (Perception.CAMERA, Perception.BOTH):
                self.camera_measurements_stamped_nonrt = None
                # Setup subscriber to camera_measurements topic
                camera_meas_topic = 'camera_measurements'
                rospy.Subscriber(
                    camera_meas_topic,
                    crowd_navigation_msgs.msg.MeasurementsSetStamped,
                    self.camera_measurements_callback
                )

        # Setup publisher to crowd_motion_prediction topic
        crowd_prediction_topic = 'crowd_motion_prediction'
        self.crowd_motion_prediction_publisher = rospy.Publisher(
            crowd_prediction_topic,
            crowd_navigation_msgs.msg.CrowdMotionPredictionStamped,
            queue_size=1
        )

        # Setup publisher to measurements topic
        measurements_topic = 'measurements'
        self.measurements_publisher = rospy.Publisher(
            measurements_topic,
            crowd_navigation_msgs.msg.MeasurementsSetStamped,
            queue_size=1
        )

    def laser_measurements_callback(self, msg):
        with self.data_lock:
            self.laser_measurements_stamped_nonrt = MeasurementsSetStamped.from_message(msg)

    def camera_measurements_callback(self, msg):
        with self.data_lock:
            self.camera_measurements_stamped_nonrt = MeasurementsSetStamped.from_message(msg)

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
            with self.data_lock:
                self.agents_pos_nonrt = agents_pos

    def adapt_measurements_format(self, measurements_set_stamped):
        measurements_set = measurements_set_stamped.measurements_set
        measurements = np.zeros((measurements_set.size, 2))
        for i in range(measurements_set.size):
            measurements[i] = np.array([measurements_set.measurements[i].x,
                                        measurements_set.measurements[i].y])

        return measurements

    def unique_measurements(self, laser_meas, camera_meas):
        if len(laser_meas) == 0 and len(camera_meas)== 0:
            return np.array([])
        elif len(laser_meas) == 0:
            return camera_meas
        elif len(camera_meas) == 0:
            return laser_meas
        
        distance_matrix = cdist(laser_meas, camera_meas) # [n_laser x n_camera]

        laser_min_indices = np.argmin(distance_matrix, axis=1)
        camera_min_indices = np.argmin(distance_matrix, axis=0)

        mutual_correspondences = [(i, laser_min_indices[i]) for i in range(len(laser_meas)) \
                                  if (camera_min_indices[laser_min_indices[i]] == i) and (distance_matrix[i, laser_min_indices[i]] <= self.hparams.ds_cbf)]
        correspondence_indices_camera = set([j for _, j in mutual_correspondences])
        correspondence_indices_laser = set([i for i, _ in mutual_correspondences])

        # take camera measurements only from correspondences
        # corresponded_meas = camera_meas[list(correspondence_indices_camera)]
        # take laser measurements only from correspondences
        # corresponded_meas = laser_meas[list(correspondence_indices_laser)]
        # take the mean of both measurements from correspondences
        corresponded_meas = [np.array([(laser_meas[i, 0] + camera_meas[j, 0]) / 2,
                                       (laser_meas[i, 1] + camera_meas[j, 1]) / 2]) for i, j in mutual_correspondences]
        if corresponded_meas:
            corresponded_meas = np.array(corresponded_meas)
        else:
            corresponded_meas = np.empty((0, 2))

        remaining_laser_indices = set(range(len(laser_meas))) - correspondence_indices_laser
        remaining_camera_indices = set(range(len(camera_meas))) - correspondence_indices_camera
        
        remaining_laser_meas = laser_meas[list(remaining_laser_indices)]
        remaining_camera_meas = camera_meas[list(remaining_camera_indices)]
        
        unique_meas = np.vstack((corresponded_meas, remaining_laser_meas, remaining_camera_meas))
        return unique_meas

    def log_values(self):
        output_dict = {}
        output_dict['cpu_time'] = self.time_history
        output_dict['kfs'] = self.kalman_infos
        output_dict['frequency'] = self.hparams.predictor_frequency
        output_dict['use_kalman'] = self.hparams.use_kalman
        if self.hparams.use_kalman:
            output_dict['associations'] = self.associations
            output_dict['estimations'] = self.estimation_history
        
        # log the data in a .json file
        log_dir = self.hparams.log_dir
        filename = self.hparams.predictor_file
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_path = os.path.join(log_dir, filename)
        with open(log_path, 'w') as file:
            json.dump(output_dict, file)

    def run(self):
        if self.hparams.n_filters == 0:
            rospy.logwarn("Crowd prediction disabled")
            return

        rate = rospy.Rate(self.hparams.predictor_frequency)
        N = self.hparams.N_horizon
        
        if self.hparams.use_kalman:
            fsms = [FSM(self.hparams) for _ in range(self.hparams.n_filters)]

        if self.hparams.log:
            rospy.on_shutdown(self.log_values)

        while not rospy.is_shutdown():
            start_time = time.time()

            if self.hparams.perception == Perception.GTRUTH and self.agents_pos_nonrt is None:
                rospy.logwarn("Missing ground truth")
                rate.sleep()
                continue
            if (self.hparams.perception == Perception.BOTH) and \
               (self.laser_measurements_stamped_nonrt is None or self.camera_measurements_stamped_nonrt is None):
                rospy.logwarn("Missing measurements")
                rate.sleep()
                continue
            if self.hparams.perception == Perception.LASER and self.laser_measurements_stamped_nonrt is None:
                rospy.logwarn("Missing laser measurements")
                rate.sleep()
                continue
            if self.hparams.perception == Perception.CAMERA and self.camera_measurements_stamped_nonrt is None:
                rospy.logwarn("Missing camera measurements")
                rate.sleep()
                continue

            # Create crowd motion prediction message (based on the perception type)
            crowd_motion_prediction = CrowdMotionPrediction()

            if self.hparams.perception == Perception.GTRUTH:
                measurements = self.agents_pos_nonrt
            elif self.hparams.perception == Perception.LASER:
                with self.data_lock:
                    measurements = self.laser_measurements_stamped_nonrt
                measurements = self.adapt_measurements_format(measurements)
            elif self.hparams.perception == Perception.CAMERA:
                with self.data_lock:    
                    measurements = self.camera_measurements_stamped_nonrt
                measurements = self.adapt_measurements_format(measurements)
            else:
                with self.data_lock:
                    laser_measurements = self.laser_measurements_stamped_nonrt
                    camera_measurements = self.camera_measurements_stamped_nonrt
                laser_measurements = self.adapt_measurements_format(laser_measurements)
                camera_measurements = self.adapt_measurements_format(camera_measurements)
                measurements = self.unique_measurements(laser_measurements, camera_measurements)

            # Create measurements message
            measurements_set = MeasurementsSet()
            for measurement in measurements:
                measurements_set.append(Measurement(measurement[0], measurement[1]))
            measurements_set_stamped = MeasurementsSetStamped(rospy.Time.from_sec(start_time),
                                                              'map',
                                                              measurements_set)
            measurements_set_stamped_msg = MeasurementsSetStamped.to_message(measurements_set_stamped)
            self.measurements_publisher.publish(measurements_set_stamped_msg)        

            if self.hparams.use_kalman:
                # Perform data association
                # predict next positions according to fsms
                next_predicted_positions = np.zeros((self.hparams.n_filters, 2))
                next_positions_cov = np.zeros((self.hparams.n_filters, 2, 2))

                for (i, fsm) in enumerate(fsms):
                    fsm_next_state = fsm.next_state
                    if self.hparams.log:
                        self.kalman_infos['KF_{}'.format(i + 1)].append([FSMStates.print(fsm.state),
                                                                         FSMStates.print(fsm_next_state),
                                                                         start_time])
                    if fsm_next_state in (FSMStates.ACTIVE, FSMStates.HOLD):
                        predicted_position = predict_next_position(fsm.current_estimate, self.hparams.dt)
                        predicted_cov = fsm.kalman_f.Pk
                    elif fsm_next_state is FSMStates.START:
                        predicted_position = fsm.current_estimate[:2]
                        predicted_cov = np.eye(4) * self.hparams.init_cov
                    else:
                        predicted_position = self.hparams.nullstate[:2]
                        predicted_cov = np.eye(4) * self.hparams.init_cov

                    next_predicted_positions[i] = predicted_position
                    next_positions_cov[i] = predicted_cov[:2, :2]

                fsm_indices = data_association(next_predicted_positions, next_positions_cov, measurements)
                if self.hparams.log:
                    self.associations.append([fsm_indices.tolist(), start_time])

                agents_estimation = np.zeros((self.hparams.n_filters, 4))
                used_measurements = np.full(measurements.shape[0], False)

                # First, update FSMs with associated measurements
                for i, fsm in enumerate(fsms):
                    # Find if the FSM has an associated measurement
                    associated_indices = np.where(fsm_indices == i)[0]
                    
                    if associated_indices.size > 0:
                        j = associated_indices[0]
                        measure = measurements[j]
                        fsm.update(start_time, measure)
                        used_measurements[j] = True
                    else:
                        # If no associated measurement, find an unassociated one
                        unassociated_indices = np.where((fsm_indices == -1) & (~used_measurements))[0]
                        
                        if unassociated_indices.size > 0:
                            k = unassociated_indices[0]
                            measure = measurements[k]
                            fsm.update(start_time, measure)
                            fsm_indices[k] = i
                            used_measurements[k] = True
                        else:
                            # No measurements available, update with None
                            fsm.update(start_time, None)

                    # Update agents_estimation with the current estimate of the FSM
                    current_estimate = fsm.current_estimate
                    agents_estimation[i] = current_estimate

                    # Append to crowd_motion_prediction
                    crowd_motion_prediction.append(MotionPrediction(Position(current_estimate[0],
                                                                                current_estimate[1],
                                                                                0.0),
                                                                    Velocity(current_estimate[2],
                                                                                current_estimate[3],
                                                                                0.0)))
            else:
                for i in range(self.hparams.n_filters):
                    if i < measurements.shape[0]:
                        current_estimate = np.array([measurements[i, 0],
                                                        measurements[i, 1],
                                                        0.0,
                                                        0.0])
                    else:
                        current_estimate = self.hparams.nullstate
                
                    crowd_motion_prediction.append(MotionPrediction(Position(current_estimate[0],
                                                                                current_estimate[1],
                                                                                0.0),
                                                                    Velocity(current_estimate[2],
                                                                                current_estimate[3],
                                                                                0.0)))

            crowd_motion_prediction_stamped = CrowdMotionPredictionStamped(rospy.Time.from_sec(start_time),
                                                                           'map',
                                                                           crowd_motion_prediction)
            crowd_motion_prediction_stamped_msg = CrowdMotionPredictionStamped.to_message(crowd_motion_prediction_stamped)
            self.crowd_motion_prediction_publisher.publish(crowd_motion_prediction_stamped_msg)
            
            # Update logged data
            if self.hparams.log:
                if self.hparams.use_kalman:
                    self.estimation_history.append(agents_estimation.tolist())

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