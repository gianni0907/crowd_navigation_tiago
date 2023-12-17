import numpy as np
import rospy
import threading

from my_tiago_controller.utils import *
from my_tiago_controller.Hparams import *
from my_tiago_controller.Status import *

import my_tiago_msgs.srv
import my_tiago_msgs.msg

class CrowdPredictionManager:
    '''
    From the laser scans input predict the motion of the humans
    Simplified case: assume to know the actual humans trajectory, 
                     only extract positions and velocities within the control horizon
                     and send them to the ControllerManager
    '''
    def __init__(self):
        self.data_lock = threading.Lock()
        # Set status
        self.status = Status.WAITING # WAITING for humans trajectory
        self.current = 0 # idx of the current humans position within trajectories
        self.hparams = Hparams()
        self.n_humans = self.hparams.n_obstacles
        self.N_horizon = self.hparams.N_horizon
        self.frequency = self.hparams.controller_frequency

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

    def set_humans_trajectory_request(self, request):
        if self.status == Status.MOVING:
            rospy.loginfo("Cannot set humans trajectory, humans are already moving")
            return my_tiago_msgs.srv.SetDesiredTargetPositionResponse(False)
        else:
            self.trajectories = CrowdMotionPrediction.from_message(request.trajectories)
            self.status = Status.MOVING
            self.current = 0
            rospy.loginfo("Humans trajectory successfully set")
            return my_tiago_msgs.srv.SetHumansTrajectoryResponse(True)
        
    def start(self):
        rate = rospy.Rate(self.frequency)

        while not rospy.is_shutdown():
            time = rospy.get_time()
            if self.status == Status.WAITING:
                rospy.logwarn("Missing humans info")
                rate.sleep()
                continue

            # Reset crowd_motion_prediction message
            crowd_motion_prediction = CrowdMotionPrediction()
            self.trajectory_length = len(self.trajectories.motion_predictions[0].positions)
            
            for i in range(self.n_humans):
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