import numpy as np

from crowd_navigation_core.Kalman import *
from crowd_navigation_core.Hparams import *
from crowd_navigation_core.utils import *

class FSM():
    '''
    Finite State Machine implementation
    4 states: idle, start, active, hold
    '''

    def __init__(self, hparams):
        self.hparams = hparams
        self.current_estimate = self.hparams.nullstate
        self.previous_estimate = self.hparams.nullstate
        self.innovation_threshold = self.hparams.innovation_threshold
        
        self.dt = self.hparams.dt
        self.kalman_f = None
        self.T_bar = self.hparams.max_pred_time
        self.next_state = FSMStates.IDLE
        self.state = FSMStates.IDLE
        self.last_valid_measure = self.hparams.nullstate[:2]
        self.last_valid_time = 0
        self.reset = False

    def idle_state(self, time, measure):
        if self.reset == True:
            self.reset = False
            self.state = FSMStates.IDLE
            self.last_valid_measure = self.hparams.nullstate[:2]
            self.last_valid_time = time
            self.current_estimate = self.hparams.nullstate
            self.previous_estimate = self.hparams.nullstate
            estimate = self.current_estimate
            
        if measure is not None:
            estimate = np.array([measure[0], measure[1], 0.0, 0.0])
            self.next_state = FSMStates.START
        else:
            estimate = self.previous_estimate
            self.next_state = FSMStates.IDLE

        return estimate
    
    def start_state(self, time, measure):
        if measure is not None:
            estimate = np.array([measure[0],
                                 measure[1],
                                 0.0,
                                 0.0])
            self.next_state = FSMStates.ACTIVE
            # Kalman Filter initialization
            self.kalman_f = Kalman(estimate, time, print_info=False)
        else:
            estimate = self.previous_estimate
            self.next_state = FSMStates.IDLE
            self.reset = True

        return estimate
    
    def active_state(self, time, measure):
        if measure is not None:
            self.kalman_f.predict(time)
            _, innovation = self.kalman_f.correct(measure)
            if np.linalg.norm(innovation) < self.innovation_threshold:
                estimate = self.kalman_f.X_k
                self.next_state = FSMStates.ACTIVE
            else:
                print(f"Reset by innovation={innovation}, {time}")
                estimate = np.array([measure[0],
                                     measure[1],
                                     0.0,
                                     0.0])
                self.next_state = FSMStates.START
        else:
            self.kalman_f.predict(time)
            estimate, _ = self.kalman_f.correct(self.last_valid_measure)
            self.next_state = FSMStates.HOLD

        return estimate
    
    def hold_state(self, time, measure):
        if measure is not None:
            self.kalman_f.predict(time)
            _, innovation = self.kalman_f.correct(measure)
            if np.linalg.norm(innovation) < self.innovation_threshold:
                estimate = self.kalman_f.X_k
                self.next_state = FSMStates.ACTIVE
            else:
                print(f"Reset by innovation={innovation}, {time}")
                estimate = np.array([measure[0],
                                     measure[1],
                                     0.0,
                                     0.0])
                self.next_state = FSMStates.START
        else:
            if time <= self.last_valid_time + self.T_bar:
                self.kalman_f.predict(time)
                estimate, _ = self.kalman_f.correct(self.last_valid_measure)
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

        if measure is not None:
            self.last_valid_measure = measure
            self.last_valid_time = time

        return self.current_estimate