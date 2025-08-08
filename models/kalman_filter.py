import numpy as np
import logging

class KalmanFilter:
    """Kalman filter for object tracking"""
    
    def __init__(self, dt=1.0, process_noise=1e-2, measurement_noise=1e-1):
        """
        Initialize Kalman filter
        
        Args:
            dt: Time step
            process_noise: Process noise covariance
            measurement_noise: Measurement noise covariance
        """
        self.dt = dt
        
        # State vector: [x, y, vx, vy] (position and velocity)
        self.state = np.zeros(4)
        
        # State covariance matrix
        self.P = np.eye(4) * 1000  # Initial uncertainty
        
        # State transition matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * process_noise
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_noise
        
        logging.debug("Kalman filter initialized")
    
    def predict(self):
        """
        Predict next state
        
        Returns:
            Predicted state vector
        """
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state
    
    def update(self, measurement):
        """
        Update state with measurement
        
        Args:
            measurement: Measurement vector [x, y]
        """
        # Convert measurement to numpy array and ensure it's 1D with 2 elements
        measurement = np.array(measurement).flatten()
        
        # Ensure measurement has exactly 2 elements
        if len(measurement) != 2:
            logging.warning(f"Measurement should have 2 elements, got {len(measurement)}")
            return
        
        # Calculate innovation
        y = measurement - self.H @ self.state
        
        # Calculate innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Calculate Kalman gain
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Handle singular matrix
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
    
    def get_state(self):
        """Get current state"""
        return self.state
    
    def get_position(self):
        """Get current position"""
        return self.state[:2]
    
    def get_velocity(self):
        """Get current velocity"""
        return self.state[2:]
    
    def get_speed(self):
        """Get current speed (magnitude of velocity)"""
        velocity = self.get_velocity()
        return np.linalg.norm(velocity)
    
    def set_state(self, state):
        """Set state manually"""
        self.state = np.array(state)
    
    def reset(self):
        """Reset filter to initial state"""
        self.state = np.zeros(4)
        self.P = np.eye(4) * 1000
        logging.debug("Kalman filter reset")
