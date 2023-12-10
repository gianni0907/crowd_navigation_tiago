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
            trajectories[1, :, 0],
            trajectories[1, :, 1],
            trajectories[2, :, 0],
            trajectories[2, :, 1],
            trajectories[3, :, 0],
            trajectories[3, :, 1],
            trajectories[4, :, 0],
            trajectories[4, :, 1]
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

    # Set the parameters required to create humans trajectory
    n_humans = Hparams.n_obstacles
    n_steps = Hparams.n_traj_steps
    initial_pos = Hparams.obstacles_initial_pos
    final_pos = Hparams.obstacles_final_pos
    trajectories = np.ndarray((n_humans, n_steps * 2, 2))
    
    # Create n_humans humans trajectory from init to final position and back
    for i in range(n_humans):
        trajectories[i, :n_steps, :] = linear_trajectory(initial_pos[i, :], final_pos[i, :], n_steps)
        trajectories[i, n_steps:, :] = linear_trajectory(final_pos[i, :], initial_pos[i, :], n_steps)
    
    # Service request:
    rospy.loginfo("Sending humans trajectories")
    send_humans_trajectory(trajectories)


            