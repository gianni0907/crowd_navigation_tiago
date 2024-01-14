import numpy as np
import rospy

import my_tiago_msgs.srv
from my_tiago_controller.Hparams import *
from my_tiago_controller.utils import *

def send_actors_trajectory(trajectories):
    rospy.wait_for_service('SetActorsTrajectory')

    try:
        set_actors_trajectory_service = rospy.ServiceProxy(
            'SetActorsTrajectory',
            my_tiago_msgs.srv.SetActorsTrajectory
        )

        response = set_actors_trajectory_service(trajectories)

        if response.success:
            rospy.loginfo('Actors trajectory successfully set')
        else:
            rospy.loginfo('Could not set actors trajectories')

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def main():
    rospy.init_node('send_actors_trajectory', log_level=rospy.INFO)
    rospy.loginfo('send actors trajectory module [OK]')

    # Create the actors trajectory (this will be deleted and substituted with laser scan readings)
    n_actors = Hparams.n_actors
    if n_actors == 0:
        rospy.logwarn("No actors")
        return
    
    positions_i = np.array([Position(2.0, 5.0),
                            Position(3.5, 3.0),
                            Position(3.5, -4.0),
                            Position(4.0, -1.0),
                            Position(-2.0, -3.0)]) # initial position of the actors
    positions_f = np.array([Position(2.0, -4.0),
                            Position(-4.5, 3.0),
                            Position(3.5, 4.0),
                            Position(-4.0, -1.0),
                            Position(-2.0, 5.0)]) # final position of the actors
    n_steps = 300 # number of steps to go from init to final positions (and vice-versa)
    trajectories = CrowdMotionPrediction()

    for i in range(n_actors):
        positions = [Position(0.0, 0.0) for _ in range(n_steps * 2)]
        velocities = [Velocity(0.0, 0.0) for _ in range(n_steps * 2)]
        positions[:n_steps], velocities[:n_steps] = linear_trajectory(positions_i[i], positions_f[i], n_steps)
        positions[n_steps:], velocities[n_steps:] = linear_trajectory(positions_f[i], positions_i[i], n_steps)
        trajectories.append(MotionPrediction(positions, velocities))
    
    # Service request:
    rospy.loginfo("Sending actors trajectories")
    send_actors_trajectory(CrowdMotionPrediction.to_message(trajectories))


            