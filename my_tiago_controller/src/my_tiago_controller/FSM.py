import numpy as np
import rospy

from my_tiago_controller.Kalman import *
from my_tiago_controller.Hparams import *
from my_tiago_controller.FSMStates import *

class FSM():
    '''
    Finite State Machine implementation
    4 states: idle, start, active, hold
    '''

    def __init__(self, hparams):
        self.hparams = hparams
        self.current_estimate = np.empty(4)
        self.previous_estimate = np.empty(4)
        self.innovation_threshold = self.hparams.innovation_threshold
        self.matching_threshold = self.hparams.matching_threshold
        
        self.kalman_f = None
        self.T_bar = self.hparams.max_pred_time
        self.next_state = FSMStates.IDLE
        self.state = FSMStates.IDLE
        self.last_valid_measurement = (np.zeros(2), 0)
        self.reset = False

    def idle_state(self, time, measure):
        if self.reset == True:
            self.reset = False
            self.state = FSMStates.IDLE
            self.last_valid_measurement = ([0.0, 0.0], time)
            self.current_estimate = np.empty(4)
            self.previous_estimate = np.empty(4)
            estimate = self.current_estimate
            
        if measure[0] != 0 or measure[1] != 0:
            estimate = np.array([measure[0], measure[1], 0.0, 0.0])
            self.next_state = FSMStates.START
        else:
            estimate = self.previous_estimate
            self.next_state = FSMStates.IDLE

        return estimate
    
    def start_state(self, time, measure):
        if measure[0] != 0 or measure[1] != 0:
            if np.linalg.norm(measure - self.previous_estimate[:2]) < self.matching_threshold:
                dt = time - self.last_valid_measurement[1]
                print(f"dt={dt} at instant {time}")
                estimate = np.array([measure[0],
                                     measure[1],
                                     (1 / dt) * (measure[0] - self.previous_estimate[0]),
                                     (1 / dt) * (measure[1] - self.previous_estimate[1])])
                self.next_state = FSMStates.ACTIVE
                # Kalman Filter initialization
                self.kalman_f = Kalman(estimate, time, print_info=False)
            else:
                rospy.loginfo("Mismatched measure, restarting")
                estimate = np.array([measure[0], measure[1], 0.0, 0.0])
                self.next_state = FSMStates.START
        else:
            estimate = self.previous_estimate
            self.next_state = FSMStates.IDLE
            self.reset = True

        return estimate
    
    def active_state(self, time, measure):
        if measure[0] != 0 or measure[1]!= 0:
            self.kalman_f.predict(time)
            _, innovation = self.kalman_f.correct(measure)
            if np.linalg.norm(innovation) < self.innovation_threshold:
                estimate = self.kalman_f.X_k
                self.next_state = FSMStates.ACTIVE
            else:
                print(f"Reset by innovation={innovation}")
                estimate = np.array([measure[0],
                                     measure[1],
                                     self.previous_estimate[2],
                                     self.previous_estimate[3]])
                self.next_state = FSMStates.START
        else:
            estimate = self.previous_estimate
            self.next_state = FSMStates.HOLD

        return estimate
    
    def hold_state(self, time, measure):
        if measure[0] != 0 or measure[1]!= 0:
            estimate = self.previous_estimate
            self.next_state = FSMStates.ACTIVE
        else:
            if time <= self.last_valid_measurement[1] + self.T_bar:
                estimate = self.kalman_f.predict(time)
                self.next_state = FSMStates.HOLD
            else:
                estimate = self.previous_estimate
                self.reset = True
                self.next_state = FSMStates.IDLE

        return estimate
    
    def update(self, time, measure):
        self.state = self.next_state
        self.previous_estimate = np.copy(self.current_estimate)

        if self.state == FSMStates.IDLE:
            self.current_estimate = self.idle_state(time, measure)
        elif self.state == FSMStates.START:
            self.current_estimate = self.start_state(time, measure)
        elif self.state == FSMStates.ACTIVE:
            self.current_estimate = self.active_state(time, measure)
        elif self.state == FSMStates.HOLD:
            self.current_estimate = self.hold_state(time, measure)

        if measure[0] != 0 or measure[1]!= 0:
            self.last_valid_measurement = (measure, time)

        return self.current_estimate