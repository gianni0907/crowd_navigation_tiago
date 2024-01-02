import numpy as np
import os
import json
import rospy
import threading
import sensor_msgs.msg
import tf2_ros

from my_tiago_controller.utils import *
from my_tiago_controller.Hparams import *
from my_tiago_controller.Status import *
from my_tiago_controller.FSM import *

import my_tiago_msgs.srv
import my_tiago_msgs.msg

class CrowdPredictionManager:
    '''
    From the laser scans input predict the motion of the actors
    Simplified case: assume to have the whole actors trajectory
                     and the data association, 
                     use the Kalman Filter to estimate next state
                     and project it over the control horizon,
                     then send the predicted trajectories to the ControllerManager
    '''
    def __init__(self):
        self.data_lock = threading.Lock()

        # Set status
        # 3 possibilities:
        #   WAITING for the initial robot state
        #   READY to get the actors trajectory
        #   MOVING actors
        self.status = Status.WAITING # WAITING for the initial robot state
        
        self.current = 0 # idx of the current actors position within trajectories

        self.robot_state = State(0.0, 0.0, 0.0, 0.0, 0.0)
        self.actors_position = {}
        self.hparams = Hparams()
        self.n_actors = self.hparams.n_actors
        self.actors_name = ['actor_{}'.format(i + 1) for i in range(self.n_actors)]
        self.actors_position_history = {key: list() for key in self.actors_name}
        # self.robot_state_history = []
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

        # Setup publisher for crowd motion prediction:
        crowd_prediction_topic = 'crowd_motion_prediction'
        self.crowd_motion_prediction_publisher = rospy.Publisher(
            crowd_prediction_topic,
            my_tiago_msgs.msg.CrowdMotionPredictionStamped,
            queue_size=1
        )

        # Setup ROS Service to set actors trajectories:
        self.set_actors_trajectory_srv = rospy.Service(
            'SetActorsTrajectory',
            my_tiago_msgs.srv.SetActorsTrajectory,
            self.set_actors_trajectory_request
        )

    def joint_states_callback(self, msg):
        self.wheels_vel = np.array([msg.velocity[13], msg.velocity[12]])

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
        
    def set_actors_trajectory_request(self, request):
        if self.status == Status.WAITING:
            rospy.loginfo("Cannot set actors trajectory, robot is not READY")
            return my_tiago_msgs.srv.SetDesiredTargetPositionResponse(False)            
        if self.status == Status.READY:
            self.trajectories = CrowdMotionPrediction.from_message(request.trajectories)
            self.status = Status.MOVING
            self.current = 0
            self.trajectory_length = len(self.trajectories.motion_predictions[0].positions)
            rospy.loginfo("Actors trajectory successfully set")
            return my_tiago_msgs.srv.SetActorsTrajectoryResponse(True)
        elif self.status == Status.MOVING:
            rospy.loginfo("Cannot set actors trajectory, actors are already moving")
            return my_tiago_msgs.srv.SetDesiredTargetPositionResponse(False)
        
    def update_actors_position(self):
        actors_position = {}
        for i in range(self.n_actors):
            actors_position[self.actors_name[i]] = self.trajectories.motion_predictions[i].positions[self.current]
        self.data_lock.acquire()
        self.actors_position = actors_position
        self.data_lock.release()

    def propagate_state(self, state):
        predictions = [np.empty(4) for _ in range(self.N_horizon)]
        time = 0
        dt = 1 / self.frequency
        for i in range(self.N_horizon):
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
        output_dict['actors'] = self.actors_position_history
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
        
        fsms = [FSM(self.hparams) for _ in range(self.n_actors)]

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
            
            if self.status == Status.READY:
                rospy.logwarn("Missing actors info")
                rate.sleep()
                continue
            
            if self.status == Status.MOVING:
                self.update_actors_position()
                if self.hparams.log:
                    for actor_name in self.actors_position.keys():
                        self.actors_position_history[actor_name].append([self.actors_position[actor_name].x,
                                                                         self.actors_position[actor_name].y,
                                                                         time])

                # Reset crowd_motion_prediction message
                crowd_motion_prediction = CrowdMotionPrediction()

                for i,fsm in enumerate(fsms):
                    fsm_state = fsm.next_state

                    if self.hparams.log:
                        self.kalman_infos['KF_{}'.format(i + 1)].append([FSMStates.print(fsm.state),
                                                                         FSMStates.print(fsm_state),
                                                                         time])

                    measure = np.array([self.actors_position[self.actors_name[i]].x,
                                        self.actors_position[self.actors_name[i]].y])
                    fsm.update(time, measure)

                    current_estimate = fsm.current_estimate
                    predictions = self.propagate_state(current_estimate)
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
                self.current += 1
                if self.current == self.trajectory_length:
                    self.status = Status.READY            

            rate.sleep()

def main():
    rospy.init_node('tiago_crowd_prediction', log_level=rospy.INFO)
    rospy.loginfo('TIAGo crowd prediction module [OK]')

    crowd_prediction_manager = CrowdPredictionManager()
    crowd_prediction_manager.run()