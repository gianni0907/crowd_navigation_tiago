import numpy as np
import rospy
import threading
import tf2_ros

from my_tiago_controller.utils import *
from my_tiago_controller.Hparams import *
from my_tiago_controller.Status import *
from my_tiago_controller.FSM import *

import my_tiago_msgs.srv
import my_tiago_msgs.msg

class CrowdPredictionManager:
    '''
    From the laser scans input predict the motion of the humans
    Simplified case: assume to have the whole humans trajectory
                     and the data association, 
                     use the Kalman Filter to estimate next state
                     and project it over the control horizon,
                     then send the predicted trajectories to the ControllerManager
    '''
    def __init__(self):
        self.data_lock = threading.Lock()
        # Set status
        self.status = Status.WAITING # WAITING for humans trajectory
        self.current = 0 # idx of the current humans position within trajectories

        self.robot_configuration = Configuration(0.0, 0.0, 0.0)
        self.actors_state = {}
        self.hparams = Hparams()
        self.n_actors = self.hparams.n_obstacles
        self.previous_time = 0.0
        self.actors_name = ['actor_{}'.format(i + 1) for i in range(self.n_actors)]
        self.actors_state_history = {key: list() for key in self.actors_name}
        self.robot_configuration_history = []
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

        # Setup publisher for crowd motion prediction:
        crowd_prediction_topic = 'crowd_motion_prediction'
        self.crowd_motion_prediction_publisher = rospy.Publisher(
            crowd_prediction_topic,
            my_tiago_msgs.msg.CrowdMotionPredictionStamped,
            queue_size=1
        )

        # Setup ROS Service to set humans trajectories:
        self.set_humans_trajectory_srv = rospy.Service(
            'SetHumansTrajectory',
            my_tiago_msgs.srv.SetHumansTrajectory,
            self.set_humans_trajectory_request
        )

    def set_from_tf_transform(self, transform):
        q = transform.transform.rotation
        theta = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        self.robot_configuration.theta = theta
        self.robot_configuration.x = transform.transform.translation.x + self.hparams.b * math.cos(theta)
        self.robot_configuration.y = transform.transform.translation.y + self.hparams.b * math.sin(theta)

    def update_configuration(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_footprint_frame, rospy.Time()
            )
            self.set_from_tf_transform(transform)
            return True
        except(tf2_ros.LookupException,
               tf2_ros.ConnectivityException,
               tf2_ros.ExtrapolationException):
            rospy.logwarn("Missing current configuration")
            return False
        
    def set_humans_trajectory_request(self, request):
        if self.status == Status.MOVING:
            rospy.loginfo("Cannot set humans trajectory, humans are already moving")
            return my_tiago_msgs.srv.SetDesiredTargetPositionResponse(False)
        else:
            self.trajectories = CrowdMotionPrediction.from_message(request.trajectories)
            self.status = Status.MOVING
            self.current = 0
            self.trajectory_length = len(self.trajectories.motion_predictions[0].positions)
            rospy.loginfo("Humans trajectory successfully set")
            return my_tiago_msgs.srv.SetHumansTrajectoryResponse(True)
        
    def update_actors_state(self):
        actors_state = {}
        for i in range(self.n_actors):
            p = self.trajectories.motion_predictions[i].positions[self.current]
            v = self.trajectories.motion_predictions[i].velocities[self.current]

        
    def run(self):
        rate = rospy.Rate(self.frequency)
        self.previous_time = rospy.get_time()
        fsms = [FSM(self.hparams) for i in range(self.n_actors)]

        while not rospy.is_shutdown():
            time = rospy.get_time()
            
            if self.status == Status.WAITING:
                if self.update_configuration():
                    self.status = Status.READY
            
            if self.status == Status.WAITING:
                rospy.logwarn("Missing humans info")
                rate.sleep()
                continue
            elif self.status == Status.MOVING:
                continue
            # Reset crowd_motion_prediction message
            crowd_motion_prediction = CrowdMotionPrediction()
            
            for i in range(self.n_actors):
                positions = [Position(0.0, 0.0) for _ in range(self.N_horizon)]
                velocities = [Velocity(0.0, 0.0) for _ in range(self.N_horizon)]
                # Create the prediction within the horizon
                if self.current + self.N_horizon <= self.trajectory_length:
                    positions = self.trajectories.motion_predictions[i].positions[self.current : self.current + self.N_horizon]
                    velocities = self.trajectories.motion_predictions[i].velocities[self.current : self.current + self.N_horizon]
                elif self.current < self.trajectory_length:
                    positions[:self.trajectory_length - self.current] = \
                        self.trajectories.motion_predictions[i].positions[self.current : self.trajectory_length]
                    velocities[:self.trajectory_length - self.current] = \
                        self.trajectories.motion_predictions[i].velocities[self.current : self.trajectory_length] 
                    for j in range(self.N_horizon - self.trajectory_length + self.current):
                        positions[self.trajectory_length - self.current + j] = \
                            self.trajectories.motion_predictions[i].positions[-1]
                        velocities[self.trajectory_length - self.current + j] = \
                            self.trajectories.motion_predictions[i].velocities[-1]
                else:
                    for j in range(self.N_horizon):
                        positions[j] = self.trajectories.motion_predictions[i].positions[-1]
                        velocities[j] = self.trajectories.motion_predictions[i].velocities[-1]
                    self.status = Status.WAITING

                crowd_motion_prediction.append(MotionPrediction(positions, velocities))

            crowd_motion_prediction_stamped = CrowdMotionPredictionStamped(rospy.Time.from_sec(time),
                                                                           'map',
                                                                           crowd_motion_prediction)
            crowd_motion_prediction_stamped_msg = CrowdMotionPredictionStamped.to_message(crowd_motion_prediction_stamped)
            self.crowd_motion_prediction_publisher.publish(crowd_motion_prediction_stamped_msg)
            self.current += 1

            rate.sleep()

def main():
    rospy.init_node('tiago_crowd_prediction', log_level=rospy.INFO)
    rospy.loginfo('TIAGo crowd prediction module [OK]')

    crowd_prediction_manager = CrowdPredictionManager()
    crowd_prediction_manager.start()