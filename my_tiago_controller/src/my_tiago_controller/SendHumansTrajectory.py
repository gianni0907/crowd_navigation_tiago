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

        response = set_humans_trajectory_service(trajectories)

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
    positions_i = np.array([Position(2.0, 2.0),
                            Position(2.0, -0.5),
                            Position(4.5, 1.0)]) # initial position of the obstacles
    positions_f = np.array([Position(1.0, -1.0),
                            Position(3.0, 3.0),
                            Position(3.0, -2.0)]) # final position of the obstacles
    n_steps = 150 # number of steps to go from init to final positions (and vice-versa)
    trajectories = CrowdMotionPrediction()

    for i in range(n_humans):
        positions = [Position(0.0, 0.0) for _ in range(n_steps * 2)]
        velocities = [Velocity(0.0, 0.0) for _ in range(n_steps * 2)]
        positions[:n_steps], velocities[:n_steps] = linear_trajectory(positions_i[i], positions_f[i], n_steps)
        positions[n_steps:], velocities[n_steps:] = linear_trajectory(positions_f[i], positions_i[i], n_steps)
        trajectories.append(MotionPrediction(positions, velocities))
    
    if n_humans == 0:
        rospy.logwarn("No humans")
        return
    # Service request:
    rospy.loginfo("Sending humans trajectories")
    send_humans_trajectory(CrowdMotionPrediction.to_message(trajectories))


            