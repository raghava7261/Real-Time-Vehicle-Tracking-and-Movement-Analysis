import numpy as np
import cv2
from collections import OrderedDict
from models.kalman_filter import KalmanFilter
import logging

class MultiObjectTracker:
    """Multi-object tracker using Kalman filters"""
    
    def __init__(self, max_disappeared=20, max_distance=100):
        """
        Initialize multi-object tracker
        
        Args:
            max_disappeared: Maximum frames a track can be missing before deletion
            max_distance: Maximum distance for associating detections to tracks
        """
        self.next_id = 0
        self.tracks = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
        logging.info("Multi-object tracker initialized")
    
    def register_track(self, centroid, detection):
        """Register a new track"""
        # Ensure centroid is 2D [x, y]
        centroid = np.array(centroid).flatten()
        if len(centroid) < 2:
            logging.warning(f"Invalid centroid for track registration: {centroid}")
            return
        
        centroid_2d = centroid[:2].astype(float)  # Take only x, y coordinates
        
        # Create Kalman filter for this track
        kalman_filter = KalmanFilter()
        kalman_filter.predict()
        kalman_filter.update(centroid_2d)
        
        self.tracks[self.next_id] = {
            'kalman_filter': kalman_filter,
            'centroid': centroid_2d,
            'detection': detection,
            'trajectory': [centroid_2d],
            'age': 0,
            'total_visible_count': 1,
            'consecutive_invisible_count': 0
        }
        
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        
        logging.debug(f"Registered new track with ID: {self.next_id - 1}")
    
    def deregister_track(self, track_id):
        """Remove a track"""
        if track_id in self.tracks:
            del self.tracks[track_id]
            del self.disappeared[track_id]
            logging.debug(f"Deregistered track with ID: {track_id}")
    
    def update(self, detections):
        """
        Update tracker with new detections
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary of active tracks
        """
        # If no detections, mark all tracks as disappeared
        if len(detections) == 0:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                self.tracks[track_id]['consecutive_invisible_count'] += 1
                
                # Remove tracks that have been missing too long
                if self.disappeared[track_id] > self.max_disappeared:
                    self.deregister_track(track_id)
            
            return self.get_active_tracks()
        
        # Initialize input centroids - ensure they are 2D points
        input_centroids = []
        for det in detections:
            if 'center' in det:
                center = det['center']
                # Ensure center is [x, y] format
                if isinstance(center, (list, tuple, np.ndarray)):
                    center = np.array(center).flatten()[:2]  # Take only x, y
                    input_centroids.append(center)
        
        input_centroids = np.array(input_centroids) if input_centroids else np.array([]).reshape(0, 2)
        
        # If no existing tracks, register all detections as new tracks
        if len(self.tracks) == 0:
            for i, detection in enumerate(detections):
                if i < len(input_centroids):
                    self.register_track(input_centroids[i], detection)
        else:
            # Get current track centroids (predicted positions)
            track_centroids = []
            track_ids = []
            
            for track_id, track in self.tracks.items():
                # Predict next position using Kalman filter
                predicted_position = track['kalman_filter'].predict()
                track_centroids.append(predicted_position[:2])  # x, y coordinates
                track_ids.append(track_id)
            
            track_centroids = np.array(track_centroids)
            
            # Compute distance matrix between detections and tracks
            distances = self._compute_distances(input_centroids, track_centroids)
            
            # Associate detections to tracks using Hungarian algorithm approximation
            matched_indices = self._associate_detections_to_tracks(distances)
            
            # Update matched tracks
            for detection_idx, track_idx in matched_indices:
                if (track_idx is not None and detection_idx is not None and 
                    track_idx < len(track_ids) and detection_idx < len(input_centroids)):
                    
                    track_id = track_ids[track_idx]
                    centroid = input_centroids[detection_idx]
                    
                    # Ensure centroid is exactly 2D [x, y]
                    if len(centroid) >= 2:
                        centroid_2d = centroid[:2].astype(float)  # Ensure only x, y coordinates
                        self.tracks[track_id]['kalman_filter'].update(centroid_2d)
                        
                        # Update track information
                        self.tracks[track_id]['centroid'] = centroid_2d
                        self.tracks[track_id]['detection'] = detections[detection_idx]
                        self.tracks[track_id]['trajectory'].append(centroid_2d)
                        self.tracks[track_id]['age'] += 1
                        self.tracks[track_id]['total_visible_count'] += 1
                        self.tracks[track_id]['consecutive_invisible_count'] = 0
                        
                        # Reset disappeared counter
                        self.disappeared[track_id] = 0
                        
                        # Limit trajectory length
                        if len(self.tracks[track_id]['trajectory']) > 30:
                            self.tracks[track_id]['trajectory'] = \
                                self.tracks[track_id]['trajectory'][-30:]
            
            # Handle unmatched detections (new tracks)
            unmatched_detections = [detection_idx for detection_idx, track_idx in matched_indices 
                                  if track_idx is None and detection_idx is not None]
            
            for detection_idx in unmatched_detections:
                if detection_idx < len(input_centroids):
                    self.register_track(input_centroids[detection_idx], detections[detection_idx])
            
            # Handle unmatched tracks (disappeared tracks)
            unmatched_tracks = [track_ids[track_idx] for detection_idx, track_idx in matched_indices 
                              if detection_idx is None and track_idx is not None and track_idx < len(track_ids)]
            
            for track_id in unmatched_tracks:
                self.disappeared[track_id] += 1
                self.tracks[track_id]['consecutive_invisible_count'] += 1
                
                if self.disappeared[track_id] > self.max_disappeared:
                    self.deregister_track(track_id)
        
        return self.get_active_tracks()
    
    def _compute_distances(self, detections, tracks):
        """Compute distance matrix between detections and tracks"""
        distances = np.zeros((len(detections), len(tracks)))
        
        for i, detection in enumerate(detections):
            for j, track in enumerate(tracks):
                distances[i, j] = np.linalg.norm(detection - track)
        
        return distances
    
    def _associate_detections_to_tracks(self, distances):
        """Simple greedy association (approximation of Hungarian algorithm)"""
        matched_indices = []
        
        # Handle empty distance matrix
        if distances.size == 0:
            return matched_indices
        
        # Create copies for manipulation
        detection_indices = list(range(len(distances)))
        track_indices = list(range(len(distances[0]))) if len(distances) > 0 else []
        
        # Greedy matching
        while len(detection_indices) > 0 and len(track_indices) > 0:
            # Find minimum distance
            min_distance = float('inf')
            min_detection_idx = None
            min_track_idx = None
            
            for i in detection_indices:
                for j in track_indices:
                    if distances[i, j] < min_distance:
                        min_distance = distances[i, j]
                        min_detection_idx = i
                        min_track_idx = j
            
            # If minimum distance is too large, break
            if min_distance > self.max_distance:
                break
            
            # Add matched pair
            matched_indices.append((min_detection_idx, min_track_idx))
            
            # Remove matched indices
            detection_indices.remove(min_detection_idx)
            track_indices.remove(min_track_idx)
        
        # Add unmatched detections
        for detection_idx in detection_indices:
            matched_indices.append((detection_idx, None))
        
        # Add unmatched tracks
        for track_idx in track_indices:
            matched_indices.append((None, track_idx))
        
        return matched_indices
    
    def get_active_tracks(self):
        """Get currently active tracks"""
        active_tracks = {}
        
        for track_id, track in self.tracks.items():
            if self.disappeared[track_id] <= self.max_disappeared:
                active_tracks[track_id] = {
                    'id': track_id,
                    'centroid': track['centroid'],
                    'detection': track['detection'],
                    'trajectory': track['trajectory'],
                    'age': track['age'],
                    'bbox': track['detection']['bbox']
                }
        
        return active_tracks
    
    def draw_tracks(self, frame, tracks):
        """Draw tracks on frame"""
        output_frame = frame.copy()
        
        for track_id, track in tracks.items():
            # Draw trajectory
            trajectory = track['trajectory']
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    cv2.line(output_frame, 
                           tuple(map(int, trajectory[i-1])), 
                           tuple(map(int, trajectory[i])), 
                           (0, 255, 255), 2)
            
            # Draw current position
            centroid = track['centroid']
            cv2.circle(output_frame, tuple(map(int, centroid)), 5, (0, 255, 255), -1)
            
            # Draw track ID
            cv2.putText(output_frame, f"ID: {track_id}", 
                       (int(centroid[0]), int(centroid[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        return output_frame
