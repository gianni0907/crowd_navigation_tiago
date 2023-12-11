import numpy as np
import rospy

import my_tiago_msgs.srv
from my_tiago_controller.Hparams import *
from my_tiago_controller.utils import *

def send_humans_trajectory(trajectories):
    rospy.wait_for_service('SetHumansTrajectory')

    try:
        set_humans_trajectory_service = rospy.ServiceProxy(
            'SetHumansTrajectory',
            my_tiago_msgs.srv.SetHumansTrajectory
        )

        response = set_humans_trajectory_service(
            trajectories[0, :, 0],
            trajectories[0, :, 1],
            trajectories[0, :, 2],
            trajectories[0, :, 3],
            trajectories[1, :, 0],
            trajectories[1, :, 1],
            trajectories[1, :, 2],
            trajectories[1, :, 3],
            trajectories[2, :, 0],
            trajectories[2, :, 1],
            trajectories[2, :, 2],
            trajectories[2, :, 3],
            trajectories[3, :, 0],
            trajectories[3, :, 1],
            trajectories[3, :, 2],
            trajectories[3, :, 3],
            trajectories[4, :, 0],
            trajectories[4, :, 1],
            trajectories[4, :, 2],
            trajectories[4, :, 3]
        )

        if response.success:
            rospy.loginfo('Humans trajectory successfully set')
        else:
            rospy.loginfo('Could not set humans trajectories')

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def main():
    rospy.init_node('send_humans_trajectory', log_level=rospy.INFO)
    rospy.loginfo('send humans trajectory module [OK]')

    # Create the humans trajectory (this will be deleted and substituted with laser scan readings)
    n_humans = Hparams.n_obstacles
    positions_i = np.array([[2.0, 2.0],
                            [2.0, -0.5],
                            [-1.0, 2.3],
                            [-2.0, -1.0],
                            [4.5, 1.0]]) # initial position of the obstacles
    positions_f = np.array([[1.0, -1.0],
                            [3.0, 3.0],
                            [-1.5, -2.0],
                            [2.0, -2.0],
                            [3.0, -2.0]]) # final position of the obstacles
    n_steps = 400 # number of steps to go from init to final positions (and vice-versa)
    trajectories = np.ndarray((n_humans, n_steps * 2, 4))
    for i in range(n_humans):
        trajectories[i, :n_steps, :] = linear_trajectory(positions_i[i, :], positions_f[i, :], n_steps)
        trajectories[i, n_steps:, :] = linear_trajectory(positions_f[i, :], positions_i[i, :], n_steps)
    
    # Service request:
    rospy.loginfo("Sending humans trajectories")
    send_humans_trajectory(trajectories)


            