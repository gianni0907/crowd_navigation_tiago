import numpy as np
import rospy

import crowd_navigation_msgs.srv
from crowd_navigation_core.Hparams import *
from crowd_navigation_core.utils import *

def send_agents_trajectory(trajectories):
    rospy.wait_for_service('SetAgentsTrajectory')

    try:
        set_agents_trajectory_service = rospy.ServiceProxy(
            'SetAgentsTrajectory',
            crowd_navigation_msgs.srv.SetAgentsTrajectory
        )

        response = set_agents_trajectory_service(trajectories)

        if response.success:
            rospy.loginfo('Agents trajectory successfully set')
        else:
            rospy.loginfo('Could not set agents trajectories')

    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def main():
    rospy.init_node('send_agents_trajectory', log_level=rospy.INFO)
    rospy.loginfo('send agents trajectory module [OK]')

    # Create the agents trajectory (this will be deleted and substituted with laser scan readings)
    n_agents = Hparams.n_agents
    if n_agents == 0:
        rospy.logwarn("No agents")
        return
    
    positions_i = np.array([Position(7.4, 10.5),
                            Position(9.2, -3.0),
                            Position(-3.5, 7.4),
                            Position(-1.0, 3.0),
                            Position(-1.0, 4.5)]) # initial position of the agents
    positions_f = np.array([Position(7.4, -0.5),
                            Position(9.2, 11.5),
                            Position(10.5, 7.4),
                            Position(10.0, 3.0),
                            Position(11.0, 4.5)]) # final position of the agents
    n_steps = 125 # number of steps to go from init to final positions (and vice-versa)
    trajectories = CrowdMotionPrediction()

    for i in range(n_agents):
        positions = [Position(0.0, 0.0) for _ in range(n_steps * 2)]
        velocities = [Velocity(0.0, 0.0) for _ in range(n_steps * 2)]
        positions[:n_steps], velocities[:n_steps] = linear_trajectory(positions_i[i], positions_f[i], n_steps)
        positions[n_steps:], velocities[n_steps:] = linear_trajectory(positions_f[i], positions_i[i], n_steps)
        trajectories.append(MotionPrediction(positions))
    
    # Service request:
    rospy.loginfo("Sending agents trajectories")
    send_agents_trajectory(CrowdMotionPrediction.to_message(trajectories))


            