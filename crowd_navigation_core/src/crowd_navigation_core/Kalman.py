import numpy as np

class Kalman:
    '''
    Implementation of the EKF
    Vk: covariance matrix of the state
    Wk: covariance matrix of the output
    Pk: corrected covariance
    X_k: state of the agent at k-th instant
    '''

    def __init__(self,
                 init_state,
                 init_time,
                 print_info = False):
        self.print_info = print_info
        self.X_k = np.array(init_state).T
        self.t_start = init_time
        
        self.Pk = np.eye(4) * 1e-2
        var_v = 0.01
        self.Vk = np.eye(4) * var_v
        var_w = 1
        self.Wk = np.eye(2) * var_w

        if self.print_info:
            print(f"State covariance: {self.Vk}")
            print(f"Output covariance: {self.Wk}")
    
    def predict(self, time):
        dt = time - self.t_start
        self.t_start = time
        F = np.array([[1.0, 0.0, dt, 0.0],
                      [0.0, 1.0, 0.0, dt],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        self.X_kp1_pred = np.matmul(F, self.X_k)
        self.P_kp1_pred = np.matmul(F, np.matmul(self.Pk, F.T)) + self.Vk

        if self.print_info:
            print(f"Predicted state: {self.X_kp1_pred}")
            print(f"Predicted covariance: {self.P_kp1_pred}")
            print(f"dt: {dt}")
 
        self.X_k = self.X_kp1_pred
        self.Pk = self.P_kp1_pred

        return self.X_kp1_pred
    
    def correct(self, z):
        H = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])
        innovation = z - np.matmul(H, self.X_kp1_pred)

        R = np.matmul(
                    np.matmul(self.P_kp1_pred, H.T), 
                    np.linalg.inv(np.matmul(H, np.matmul(self.P_kp1_pred, H.T)) + self.Wk)
                )
        
        if self.print_info:
            print(f"measured state: {z}")
            print(f"innovation: {innovation}")
            print(f"kalman gain: {R}")

        X_kp1 = self.X_kp1_pred + np.matmul(R, innovation)
        P_kp1 = self.P_kp1_pred - np.matmul(R, np.matmul(H, self.P_kp1_pred))
        self.X_k = X_kp1
        self.Pk = P_kp1

        if self.print_info:
            print(f"corrected state: {self.X_k}")
            print(f"corrected cov: {self.Pk}")
        return self.X_k, innovation