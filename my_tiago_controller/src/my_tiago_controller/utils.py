import numpy as np
import my_tiago_msgs.msg
import geometry_msgs.msg

class Configuration:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.theta)
    
    def get_q(self):
        return np.array([self.x, self.y, self.theta])

class Position:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return '({}, {})'.format(self.x, self.y)

    @staticmethod
    def to_message(position):
        return geometry_msgs.msg.Point(position.x, position.y, 0.0)
    
    @staticmethod
    def from_message(position_msg):
        return Position(position_msg.x, position_msg.y)

class Velocity:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return '({}, {})'.format(self.x, self.y)
    
    @staticmethod
    def to_message(velocity):
        return geometry_msgs.msg.Vector3(velocity.x, velocity.y, 0.0)
    
    @staticmethod
    def from_message(velocity_msg):
        return Velocity(velocity_msg.x, velocity_msg.y)
    
class MotionPrediction:
    def __init__(self, positions, velocities):
        self.positions = positions
        self.velocities = velocities
    
    @staticmethod
    def to_message(motion_prediction):
        positions_msg = []
        velocities_msg = []
        for i in range(len(motion_prediction.positions)):
            positions_msg.append(Position.to_message(motion_prediction.positions[i]))
            velocities_msg.append(Velocity.to_message(motion_prediction.velocities[i]))

        return my_tiago_msgs.msg.MotionPrediction(
            positions_msg,
            velocities_msg
        )

    @staticmethod
    def from_message(motion_prediction_msg):
        positions = []
        velocities = []
        for i in range(len(motion_prediction_msg.positions)):
            positions.append(Position.from_message(motion_prediction_msg.positions[i]))
            velocities.append(Velocity.from_message(motion_prediction_msg.velocities[i]))

        return MotionPrediction(
            positions,
            velocities
        )
    
class CrowdMotionPrediction:
    def __init__(self):
        self.motion_predictions = []
        self.size = 0
    
    def append(self, motion_prediction):
        self.motion_predictions.append(motion_prediction)
        self.size += 1

    @staticmethod
    def to_message(crowd_motion_prediction):
        crowd_motion_prediction_msg = \
            my_tiago_msgs.msg.CrowdMotionPrediction()
        for motion_prediction in crowd_motion_prediction.motion_predictions:
            crowd_motion_prediction_msg.motion_predictions.append(
                MotionPrediction.to_message(motion_prediction)
            )
        return crowd_motion_prediction_msg
    
    @staticmethod
    def from_message(crowd_motion_prediction_msg):
        crowd_motion_prediction = CrowdMotionPrediction()
        for motion_prediction_msg in \
            crowd_motion_prediction_msg.motion_predictions:
            crowd_motion_prediction.append(
                MotionPrediction.from_message(motion_prediction_msg)
            )
        return crowd_motion_prediction
    
class CrowdMotionPredictionStamped:
    def __init__(self, time, frame_id, crowd_motion_prediction):
        self.time = time
        self.frame_id = frame_id
        self.crowd_motion_prediction = crowd_motion_prediction

    @staticmethod
    def to_message(crowd_motion_prediction_stamped):
        crowd_motion_prediction_stamped_msg = \
            my_tiago_msgs.msg.CrowdMotionPredictionStamped()
        crowd_motion_prediction_stamped_msg.header.stamp = \
            crowd_motion_prediction_stamped.time
        crowd_motion_prediction_stamped_msg.header.frame_id = \
            crowd_motion_prediction_stamped.frame_id
        crowd_motion_prediction_stamped_msg.crowd_motion_prediction = \
            CrowdMotionPrediction.to_message(
                crowd_motion_prediction_stamped.crowd_motion_prediction
              )
        return crowd_motion_prediction_stamped_msg

    @staticmethod
    def from_message(crowd_motion_prediction_stamped_msg):
        return CrowdMotionPredictionStamped(
            crowd_motion_prediction_stamped_msg.header.stamp,
            crowd_motion_prediction_stamped_msg.header.frame_id,
            CrowdMotionPrediction.from_message(
                crowd_motion_prediction_stamped_msg.crowd_motion_prediction
            )
        )


def Euler(f, x0, u, dt):
    return x0 + f(x0,u)*dt

def RK4(f, x0, u ,dt):
    k1 = f(x0, u)
    k2 = f(x0 + k1 * dt / 2.0, u)
    k3 = f(x0 + k2 * dt / 2.0, u)
    k4 = f(x0 + k3 * dt, u)
    yf = x0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return yf

def integrate(f, x0, u, dt, integration_method='RK4'):
    if integration_method == 'RK4':
        return RK4(f, x0, u, dt)
    else:
        return Euler(f, x0, u, dt)
    
def linear_trajectory(p_i : Position, p_f : Position, n_steps):
    """
    Generate a linear trajectory between two 2D points.

    Parameters:
    - xi: Initial point of type Position
    - xf: Final point of type Position
    - n_steps: Number of steps for the trajectory (integer)

    Returns:
    - trajectory: 2D array containing positions (array of Position)
                  and velocities (array of Velocity)
    """

    # Calculate velocity
    x_vel = (p_f.x - p_i.x) / (n_steps - 1)
    y_vel = (p_f.y - p_i.y) / (n_steps - 1)

    # Initialize the positions and velocities array
    positions = np.empty(n_steps, dtype=Position)
    velocities = np.empty(n_steps, dtype=Position)

    # Generate linear trajectory
    for i in range(n_steps):
        alpha = i / (n_steps - 1)  # Interpolation parameter
        positions[i] = Position((1 - alpha) * p_i.x + alpha * p_f.x, (1 - alpha) * p_i.y + alpha * p_f.y)
        velocities[i] = Velocity(x_vel, y_vel)
    velocities[n_steps - 1] = Velocity(0.0, 0.0)

    return positions, velocities
