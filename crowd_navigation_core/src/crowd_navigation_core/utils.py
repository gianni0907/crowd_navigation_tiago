import numpy as np
import math
import crowd_navigation_msgs.msg
import geometry_msgs.msg
from enum import Enum

class State:
    def __init__(self, x, y, theta, v, omega):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.omega = omega

    def __repr__(self):
        return '({}, {}, {}, {}, {})'.format(self.x, self.y, self.theta, self.v, self.omega)
    
    def get_state(self):
        return np.array([self.x, self.y, self.theta, self.v, self.omega])

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
    
class Measurements:
    def __init__(self):
        self.positions = []
        self.size = 0

    def append(self, core_point):
        self.positions.append(core_point)
        self.size += 1
    
    @staticmethod
    def to_message(measurements):
        measurements_msg = crowd_navigation_msgs.msg.Measurements()
        for core_point in measurements.positions:
            measurements_msg.positions.append(Position.to_message(core_point))

        return measurements_msg

    @staticmethod
    def from_message(measurements_msg):
        measurements = Measurements()
        for position_msg in measurements_msg.positions:
            measurements.append(Position.from_message(position_msg))
            
        return measurements
    
class MeasurementsStamped:
    def __init__(self, time, frame_id, measurements):
        self.time = time
        self.frame_id = frame_id
        self.measurements = measurements

    @staticmethod
    def to_message(measurements_stamped):
        measurements_stamped_msg = \
            crowd_navigation_msgs.msg.MeasurementsStamped()
        measurements_stamped_msg.header.stamp = \
            measurements_stamped.time
        measurements_stamped_msg.header.frame_id = \
            measurements_stamped.frame_id
        measurements_stamped_msg.measurements= \
            Measurements.to_message(
                measurements_stamped.measurements
            )
        return measurements_stamped_msg

    @staticmethod
    def from_message(measurements_stamped_msg):
        return MeasurementsStamped(
            measurements_stamped_msg.header.stamp,
            measurements_stamped_msg.header.frame_id,
            Measurements.from_message(
                measurements_stamped_msg.measurements
            )
        )
 
class MotionPrediction:
    def __init__(self, positions):
        self.positions = positions
    
    @staticmethod
    def to_message(motion_prediction):
        positions_msg = []
        for i in range(len(motion_prediction.positions)):
            positions_msg.append(Position.to_message(motion_prediction.positions[i]))

        return crowd_navigation_msgs.msg.MotionPrediction(positions_msg)

    @staticmethod
    def from_message(motion_prediction_msg):
        positions = []
        for i in range(len(motion_prediction_msg.positions)):
            positions.append(Position.from_message(motion_prediction_msg.positions[i]))
            
        return MotionPrediction(positions)
    
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
            crowd_navigation_msgs.msg.CrowdMotionPrediction()
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
            crowd_navigation_msgs.msg.CrowdMotionPredictionStamped()
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

class LaserScan:
    def __init__(self,
                 time,
                 frame_id,
                 angle_min, angle_max, angle_increment,
                 range_min, range_max,
                 ranges,
                 intensities):
        self.time = time
        self.frame_id = frame_id
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max
        self.ranges = ranges
        self.intensities = intensities

    @staticmethod
    def from_message(msg):
        return LaserScan(
            msg.header.stamp.to_sec(),
            msg.header.frame_id,
            msg.angle_min,
            msg.angle_max,
            msg.angle_increment,
            msg.range_min,
            msg.range_max,
            msg.ranges,
            msg.intensities
        )

class Status(Enum):
    WAITING = 0
    READY = 1
    MOVING = 2

class Perception(Enum):
    FAKE = 0
    LASER = 1
    CAMERA = 2
    BOTH = 3

    def print(mode):
        return f'{mode}'

class SelectionMode(Enum):
    CLOSEST = 0
    AVERAGE = 1

class FSMStates(Enum):
    IDLE = 0
    START = 1
    ACTIVE = 2
    HOLD = 3

    def print(state):
        return f'{state}'
    
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

def moving_average(points, window_size=1):
    smoothed_points = np.zeros(points.shape)
    
    for i in range(points.shape[0]):
        # Compute indices for the moving window
        start_idx = np.max([0, i - window_size // 2])
        end_idx = np.min([points.shape[0], i + window_size // 2 + 1])
        smoothed_points[i] = np.sum(points[start_idx : end_idx], 0) / (end_idx - start_idx)

    return smoothed_points

def z_rotation(angle, point2d):
    R = np.array([[math.cos(angle), - math.sin(angle), 0.0],
                  [math.sin(angle), math.cos(angle), 0.0],
                  [0.0, 0.0, 1.0]])
    point3d = np.array([point2d[0], point2d[1], 0.0])
    rotated_point2d = np.matmul(R, point3d)[:2]
    return rotated_point2d

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def sort_by_distance(points, reference_point):
    distances = np.array([euclidean_distance(point, reference_point) for point in points])
    sorted_indices = np.argsort(distances)
    sorted_points = points[sorted_indices]
    sorted_distances = distances[sorted_indices]
    return sorted_points, sorted_distances

# Wrap angle to [-pi, pi):
def wrap_angle(theta):
    return math.atan2(math.sin(theta), math.cos(theta))

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

def compute_normal_vector(p1, p2):
    x_p1 = p1[0]
    y_p1 = p1[1]
    x_p2 = p2[0]
    y_p2 = p2[1]
    # Compute the direction vector and its magnitude
    direction_vector = np.array([x_p2 - x_p1, y_p2 - y_p1])
    magnitude = np.linalg.norm(direction_vector)

    # Compute the normalized normal vector
    normal_vector = np.array([- direction_vector[1], direction_vector[0]])
    normalized_normal_vector = normal_vector / magnitude

    return normalized_normal_vector

def is_outside(point, vertexes, normals):
    for i, vertex in enumerate(vertexes):
        if np.dot(normals[i], point - vertex) < 0.0:
            return True
    return False

def data_association(predictions, covariances, measurements):
    n_measurements = measurements.shape[0]
    n_fsms = predictions.shape[0]

    # Heuristics parameters
    gating_tau = 10 # maximum Mahalanobis distance threshold
    gamma_threshold = 1e-1 # lonely best friends threshold

    # Initialize association info arrays
    fsm_indices = -1 * np.ones(n_measurements, dtype=int)
    distances = np.full(n_measurements, np.inf)

    if n_fsms == 0 or n_measurements == 0:
        return fsm_indices

    # Step 1: compute the association matrix
    A_mat = np.zeros((n_measurements, n_fsms))
    for i in range(n_fsms):
        info_mat = np.linalg.inv(covariances[i])
        diffs = measurements - predictions[i]
        A_mat[:, i] = np.sqrt(np.einsum('ij,ij->i', diffs @ info_mat, diffs))

    # Step 2: perform gating and initial assignment
    min_indices = np.argmin(A_mat, axis=1)
    min_distances = np.min(A_mat, axis=1)

    valid_indices = min_distances < gating_tau
    fsm_indices[valid_indices] = min_indices[valid_indices]
    distances[valid_indices] = min_distances[valid_indices]

    # Step 3: apply the best friend criterion
    for j in range(n_measurements):
        if fsm_indices[j] != -1:
            col_min = np.min(A_mat[:, fsm_indices[j]])
            if distances[j] != col_min:
                fsm_indices[j] = -1

    # Step 4: apply the lonely best friend criterion
    if n_fsms > 1 and n_measurements > 1:
        for j in range(n_measurements):
            proposed_est = fsm_indices[j]
            if proposed_est == -1:
                continue

            d_ji = distances[j]

            # find the second best value of the row
            row = A_mat[j, :]
            second_min_row = np.partition(row, 1)[1]

            # find the second best value of the col
            col = A_mat[:, proposed_est]
            second_min_col = np.partition(col, 1)[1]

            # check association ambiguity
            if (second_min_row - d_ji) < gamma_threshold or (second_min_col - d_ji) < gamma_threshold:
                fsm_indices[j] = -1
    
    return fsm_indices