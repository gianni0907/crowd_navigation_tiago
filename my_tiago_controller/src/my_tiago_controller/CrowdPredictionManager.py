import numpy as np
import rospy

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
            my_tiago_msgs.msg.CrowdMotionPrediction,
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
            self.trajectory_length = len(request.x_1)
            self.trajectories = np.ndarray((self.n_humans, self.trajectory_length, 4))
            self.trajectories[0, :, 0] = request.x_1
            self.trajectories[0, :, 1] = request.y_1
            self.trajectories[0, :, 2] = request.xdot_1
            self.trajectories[0, :, 3] = request.ydot_1
            self.trajectories[1, :, 0] = request.x_2
            self.trajectories[1, :, 1] = request.y_2
            self.trajectories[1, :, 2] = request.xdot_2
            self.trajectories[1, :, 3] = request.ydot_2
            self.trajectories[2, :, 0] = request.x_3
            self.trajectories[2, :, 1] = request.y_3
            self.trajectories[2, :, 2] = request.xdot_3
            self.trajectories[2, :, 3] = request.ydot_3
            self.trajectories[3, :, 0] = request.x_4
            self.trajectories[3, :, 1] = request.y_4
            self.trajectories[3, :, 2] = request.xdot_4
            self.trajectories[3, :, 3] = request.ydot_4
            self.trajectories[4, :, 0] = request.x_5
            self.trajectories[4, :, 1] = request.y_5
            self.trajectories[4, :, 2] = request.xdot_5
            self.trajectories[4, :, 3] = request.ydot_5
            self.status = Status.MOVING
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

            # Create the prediction within the horizon
            predictions = np.ndarray((self.n_humans, self.N_horizon, 4))
            if self.current + self.N_horizon <= self.trajectory_length:
                predictions = self.trajectories[:, self.current:self.current + self.N_horizon, :]
            else:
                predictions[:, :self.trajectory_length - self.current, :] = self.trajectories[:, self.current: self.current + self.trajectory_length, :] 
                predictions[:, self.trajectory_length - self.current:, :] = self.trajectories[:, -1, :]

            for i in range(self.n_humans):
                crowd_motion_prediction.append(
                    MotionPrediction(
                        predictions[i, : , :2],
                        predictions[i, :, 2:]
                    )
                )

            crowd_motion_prediction_stamped = CrowdMotionPredictionStamped(rospy.Time.from_sec(time),
                                                                           'map',
                                                                           crowd_motion_prediction)
            crowd_motion_prediction_stamped_msg = CrowdMotionPredictionStamped.to_message(crowd_motion_prediction_stamped)
            self.crowd_motion_prediction_publisher.publish(crowd_motion_prediction_stamped_msg)
            self.current = self.current + 1

            if self.current == self.trajectory_length:
                self.status = Status.WAITING
                self.current = 0

            rate.sleep()

def main():
    rospy.init_node('tiago_crowd_prediction', log_level=rospy.INFO)
    rospy.loginfo('TIAGo crowd prediction module [OK]')

    crowd_prediction_manager = CrowdPredictionManager()
    crowd_prediction_manager.start()