import numpy as np
import math
import crowd_navigation_msgs.msg
import geometry_msgs.msg
from enum import Enum
from numpy.linalg import norm

from shapely.geometry import Polygon, Point
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

    @staticmethod
    def set_from_tf_transform(transform):
        
        x = transform.transform.translation.x
        y = transform.transform.translation.y
        q = transform.transform.rotation
        theta = math.atan2(
          2.0 * (q.w * q.z + q.x * q.y),
          1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )
        config = Configuration(x, y, theta)
        return config

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.theta)
    
    def get_q(self):
        return np.array([self.x, self.y, self.theta])

class Position:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)

    @staticmethod
    def to_message(position):
        return geometry_msgs.msg.Point(position.x, position.y, position.z)
    
    @staticmethod
    def from_message(position_msg):
        return Position(position_msg.x, position_msg.y, position_msg.z)

class Velocity:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return '({}, {}, {})'.format(self.x, self.y, self.z)
    
    @staticmethod
    def to_message(velocity):
        return geometry_msgs.msg.Vector3(velocity.x, velocity.y, velocity.z)
    
    @staticmethod
    def from_message(velocity_msg):
        return Velocity(velocity_msg.x, velocity_msg.y, velocity_msg.z)
    
class Measurement:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return '({}, {})'.format(self.x, self.y)
    
    @staticmethod
    def to_message(meas):
        return crowd_navigation_msgs.msg.Measurement(meas.x, meas.y)
    
    @staticmethod
    def from_message(meas_msg):
        return Measurement(meas_msg.x, meas_msg.y)
class MeasurementsSet:
    def __init__(self):
        self.measurements = []
        self.size = 0

    def append(self, measurement):
        self.measurements.append(measurement)
        self.size += 1
    
    @staticmethod
    def to_message(measurements_set):
        measurements_set_msg = crowd_navigation_msgs.msg.MeasurementsSet()
        for measurement in measurements_set.measurements:
            measurements_set_msg.measurements.append(Measurement.to_message(measurement))

        return measurements_set_msg

    @staticmethod
    def from_message(measurements_set_msg):
        measurements_set = MeasurementsSet()
        for measurement_msg in measurements_set_msg.measurements:
            measurements_set.append(Measurement.from_message(measurement_msg))
            
        return measurements_set
    
class MeasurementsSetStamped:
    def __init__(self, time, frame_id, measurements_set):
        self.time = time
        self.frame_id = frame_id
        self.measurements_set = measurements_set

    @staticmethod
    def to_message(measurements_set_stamped):
        measurements_set_stamped_msg = crowd_navigation_msgs.msg.MeasurementsSetStamped()
        measurements_set_stamped_msg.header.stamp = measurements_set_stamped.time
        measurements_set_stamped_msg.header.frame_id = measurements_set_stamped.frame_id
        measurements_set_stamped_msg.measurements_set= MeasurementsSet.to_message(
                measurements_set_stamped.measurements_set
            )
        return measurements_set_stamped_msg

    @staticmethod
    def from_message(measurements_set_stamped_msg):
        return MeasurementsSetStamped(
            measurements_set_stamped_msg.header.stamp,
            measurements_set_stamped_msg.header.frame_id,
            MeasurementsSet.from_message(
                measurements_set_stamped_msg.measurements_set
            )
        )
 
class MotionPrediction:
    def __init__(self, position, velocity):
        self.position = position
        self.velocity = velocity
    
    @staticmethod
    def to_message(motion_prediction):
        return crowd_navigation_msgs.msg.MotionPrediction(
            Position.to_message(motion_prediction.position),
            Velocity.to_message(motion_prediction.velocity)
        )

    @staticmethod
    def from_message(motion_prediction_msg):
        return MotionPrediction(
            Position.from_message(motion_prediction_msg.position),
            Velocity.from_message(motion_prediction_msg.velocity)
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
        crowd_motion_prediction_msg = crowd_navigation_msgs.msg.CrowdMotionPrediction()
        for motion_prediction in crowd_motion_prediction.motion_predictions:
            crowd_motion_prediction_msg.motion_predictions.append(
                MotionPrediction.to_message(motion_prediction)
            )
        return crowd_motion_prediction_msg
    
    @staticmethod
    def from_message(crowd_motion_prediction_msg):
        crowd_motion_prediction = CrowdMotionPrediction()
        for motion_prediction_msg in crowd_motion_prediction_msg.motion_predictions:
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
        crowd_motion_prediction_stamped_msg = crowd_navigation_msgs.msg.CrowdMotionPredictionStamped()
        crowd_motion_prediction_stamped_msg.header.stamp = crowd_motion_prediction_stamped.time
        crowd_motion_prediction_stamped_msg.header.frame_id = crowd_motion_prediction_stamped.frame_id
        crowd_motion_prediction_stamped_msg.crowd_motion_prediction = CrowdMotionPrediction.to_message(
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
    GTRUTH = 0
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
    
class WorldType(Enum):
    EMPTY = 0
    TWO_ROOMS = 1
    THREE_ROOMS = 2
    CORRIDOR = 3

def predict_next_position(state, dt):
    F = np.array([[1.0, 0.0, dt, 0.0],
                  [0.0, 1.0, 0.0, dt]])
    next_position = np.matmul(F, state)

    return next_position

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

def sort_by_distance(points, reference_point):
    distances = np.array([norm(point - reference_point) for point in points])
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

def get_areas_coefficients(areas, max_vertexes):
    a_coefs = []
    b_coefs = []
    c_coefs = []
    for area in areas:
        n_vertexes = area.shape[0]
        shifted_area = np.append(area[1:n_vertexes], [area[0]], axis=0)
        delta_x = shifted_area[:n_vertexes, 0] - area[:n_vertexes, 0]
        delta_y = shifted_area[:n_vertexes, 1] - area[:n_vertexes, 1]
        norms = norm([delta_x, delta_y], axis=0)

        a = np.append(np.divide(delta_y, norms), np.zeros(max_vertexes - n_vertexes))
        b = np.append(-np.divide(delta_x, norms), np.zeros(max_vertexes - n_vertexes))
        c = np.append(np.divide(np.multiply(area[:n_vertexes, 0], shifted_area[:n_vertexes, 1]) - \
                                np.multiply(area[:n_vertexes, 1], shifted_area[:n_vertexes, 0]),
                                norms),
                      10000 * np.ones(max_vertexes - n_vertexes))

        a_coefs.append(a)
        b_coefs.append(b)
        c_coefs.append(c)
    return a_coefs, b_coefs, c_coefs

def get_area_index(areas, a_coefs, b_coefs, c_coefs, predicted_motion, threshold=0):
    area_index = None
    for idx, area in enumerate(areas):
        if is_inside_area(predicted_motion[0], a_coefs[idx], b_coefs[idx], c_coefs[idx], threshold):
            area_index = [idx] * len(predicted_motion)
    return area_index

def is_inside_areas(point, a_coefs, b_coefs, c_coefs, threshold=0.0):
    for a,b,c in zip(a_coefs, b_coefs, c_coefs):
        lhs = a * point[0] + b * point[1]
        if np.all(lhs <= c - threshold):
            return True
    return False

def is_inside_area(point, a, b, c, threshold=0):
    lhs = a * point[0] + b * point[1]
    if np.all(lhs <= c - threshold):
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
                print(f"Best friend criterion: {fsm_indices[j]}")
                fsm_indices[j] = -1

    # Step 4: apply the lonely best friend criterion
    # if n_fsms > 1 and n_measurements > 1:
    #     for j in range(n_measurements):
    #         proposed_est = fsm_indices[j]
    #         if proposed_est == -1:
    #             continue

    #         d_ji = distances[j]

    #         # find the second best value of the row
    #         row = A_mat[j, :]
    #         second_min_row = np.partition(row, 1)[1]

    #         # find the second best value of the col
    #         col = A_mat[:, proposed_est]
    #         second_min_col = np.partition(col, 1)[1]

    #         # check association ambiguity
    #         if (second_min_row - d_ji) < gamma_threshold or (second_min_col - d_ji) < gamma_threshold:
    #             print(f"Lonely best friend criterion: {proposed_est}")
    #             fsm_indices[j] = -1
    
    return fsm_indices