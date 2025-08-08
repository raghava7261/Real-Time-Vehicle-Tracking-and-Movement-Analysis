import numpy as np
import pandas as pd
from collections import defaultdict, deque
import time
import json
import logging

class Analytics:
    """Analytics and reporting for vehicle tracking system"""
    
    def __init__(self):
        """Initialize analytics system"""
        self.reset()
        logging.info("Analytics system initialized")
    
    def reset(self):
        """Reset all analytics data"""
        self.detection_history = deque(maxlen=1000)
        self.track_history = deque(maxlen=1000)
        self.vehicle_counts = defaultdict(int)
        self.total_detections = 0
        self.total_tracks = 0
        self.fps_history = deque(maxlen=30)
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.speed_estimates = deque(maxlen=100)
        self.trajectory_data = defaultdict(list)
        
        logging.info("Analytics data reset")
    
    def update(self, detections, tracks):
        """
        Update analytics with new frame data
        
        Args:
            detections: List of detection dictionaries
            tracks: Dictionary of active tracks
        """
        current_time = time.time()
        
        # Update detection history
        self.detection_history.append({
            'timestamp': current_time,
            'detections': detections,
            'count': len(detections)
        })
        
        # Update track history
        self.track_history.append({
            'timestamp': current_time,
            'tracks': tracks,
            'count': len(tracks)
        })
        
        # Update vehicle counts
        for detection in detections:
            self.vehicle_counts[detection['class']] += 1
        
        # Update totals
        self.total_detections += len(detections)
        self.total_tracks = max(self.total_tracks, len(tracks))
        
        # Calculate FPS
        if len(self.fps_history) > 0:
            time_diff = current_time - self.last_update_time
            if time_diff > 0:
                fps = 1.0 / time_diff
                self.fps_history.append(fps)
        else:
            self.fps_history.append(0)
        
        self.last_update_time = current_time
        
        # Update trajectory data and estimate speeds
        self._update_trajectory_data(tracks)
        self._estimate_speeds(tracks)
        
        # Update last update time
        self.last_update_time = current_time
    
    def _update_trajectory_data(self, tracks):
        """Update trajectory data for tracks"""
        for track_id, track in tracks.items():
            if 'trajectory' in track and len(track['trajectory']) > 0:
                self.trajectory_data[track_id] = track['trajectory']
    
    def _estimate_speeds(self, tracks):
        """Estimate speeds for tracks"""
        current_time = time.time()
        
        for track_id, track in tracks.items():
            if 'trajectory' in track and len(track['trajectory']) >= 2:
                # Get last two positions
                pos1 = np.array(track['trajectory'][-2])
                pos2 = np.array(track['trajectory'][-1])
                
                # Calculate distance (pixels per frame)
                distance = np.linalg.norm(pos2 - pos1)
                
                # Time difference between frames (assuming processing time)
                time_diff = current_time - self.last_update_time if hasattr(self, 'last_update_time') else 0.033
                time_diff = max(time_diff, 0.001)  # Prevent division by zero
                
                # Convert to speed estimate (pixels per second)
                speed_pixels_per_second = distance / time_diff
                
                # Convert to km/h (assuming 1 pixel = 0.1 meters as rough estimate)
                speed_kmh = speed_pixels_per_second * 0.1 * 3.6  # m/s to km/h
                
                # Only store if speed is reasonable (not too high)
                if speed_kmh < 200:  # Filter out unrealistic speeds
                    self.speed_estimates.append({
                        'track_id': track_id,
                        'speed_pixels_per_sec': speed_pixels_per_second,
                        'speed_kmh': speed_kmh,
                        'distance': distance,
                        'timestamp': current_time
                    })
    
    def has_data(self):
        """Check if analytics has any data"""
        return len(self.detection_history) > 0 or len(self.track_history) > 0
    
    def get_current_stats(self):
        """Get current statistics"""
        current_fps = np.mean(list(self.fps_history)) if self.fps_history else 0
        
        # Get recent detection confidences
        recent_detections = []
        if len(self.detection_history) > 0:
            recent_frame = self.detection_history[-1]
            recent_detections = recent_frame['detections']
        
        avg_confidence = 0
        if recent_detections:
            confidences = [det['confidence'] for det in recent_detections]
            avg_confidence = np.mean(confidences)
        
        active_tracks = 0
        if len(self.track_history) > 0:
            active_tracks = self.track_history[-1]['count']
        
        return {
            'total_detections': self.total_detections,
            'active_tracks': active_tracks,
            'avg_confidence': avg_confidence,
            'fps': current_fps,
            'vehicle_types': dict(self.vehicle_counts),
            'processing_time': time.time() - self.start_time
        }
    
    def get_detection_timeline(self):
        """Get detection count timeline"""
        timestamps = []
        counts = []
        
        for entry in self.detection_history:
            timestamps.append(entry['timestamp'])
            counts.append(entry['count'])
        
        return timestamps, counts
    
    def get_vehicle_distribution(self):
        """Get vehicle type distribution"""
        return dict(self.vehicle_counts)
    
    def get_speed_statistics(self):
        """Get speed statistics"""
        if not self.speed_estimates:
            return {}
        
        speeds_kmh = [est['speed_kmh'] for est in self.speed_estimates if 'speed_kmh' in est]
        speeds_pixels = [est['speed_pixels_per_sec'] for est in self.speed_estimates if 'speed_pixels_per_sec' in est]
        
        if not speeds_kmh and not speeds_pixels:
            return {}
        
        if speeds_kmh:
            return {
                'mean_speed': np.mean(speeds_kmh),
                'median_speed': np.median(speeds_kmh),
                'max_speed': np.max(speeds_kmh),
                'min_speed': np.min(speeds_kmh),
                'std_speed': np.std(speeds_kmh),
                'unit': 'km/h'
            }
        else:
            return {
                'mean_speed': np.mean(speeds_pixels),
                'median_speed': np.median(speeds_pixels),
                'max_speed': np.max(speeds_pixels),
                'min_speed': np.min(speeds_pixels),
                'std_speed': np.std(speeds_pixels),
                'unit': 'pixels/sec'
            }
    
    def get_trajectory_analysis(self):
        """Get trajectory analysis"""
        analysis = {}
        
        for track_id, trajectory in self.trajectory_data.items():
            if len(trajectory) >= 2:
                # Calculate total distance
                total_distance = 0
                for i in range(1, len(trajectory)):
                    pos1 = np.array(trajectory[i-1])
                    pos2 = np.array(trajectory[i])
                    total_distance += np.linalg.norm(pos2 - pos1)
                
                # Calculate average direction
                start_pos = np.array(trajectory[0])
                end_pos = np.array(trajectory[-1])
                direction = end_pos - start_pos
                
                analysis[track_id] = {
                    'total_distance': total_distance,
                    'direction': direction.tolist(),
                    'trajectory_length': len(trajectory)
                }
        
        return analysis
    
    def generate_report(self):
        """Generate comprehensive analytics report"""
        report = {
            'timestamp': time.time(),
            'processing_duration': time.time() - self.start_time,
            'summary': self.get_current_stats(),
            'vehicle_distribution': self.get_vehicle_distribution(),
            'speed_statistics': self.get_speed_statistics(),
            'trajectory_analysis': self.get_trajectory_analysis(),
            'detection_history_length': len(self.detection_history),
            'track_history_length': len(self.track_history)
        }
        
        return json.dumps(report, indent=2)
    
    def get_hourly_statistics(self):
        """Get hourly statistics (simulated for demo)"""
        # This would typically analyze historical data
        # For demo purposes, we'll return sample data
        hours = list(range(24))
        vehicle_counts = np.random.randint(10, 100, 24).tolist()
        
        return {
            'hours': hours,
            'vehicle_counts': vehicle_counts
        }
    
    def detect_anomalies(self):
        """Detect anomalies in traffic patterns"""
        anomalies = []
        
        # Check for unusual speed patterns
        if self.speed_estimates:
            speeds = [est['speed'] for est in self.speed_estimates]
            mean_speed = np.mean(speeds)
            std_speed = np.std(speeds)
            
            for speed_est in self.speed_estimates:
                if abs(speed_est['speed'] - mean_speed) > 2 * std_speed:
                    anomalies.append({
                        'type': 'unusual_speed',
                        'track_id': speed_est['track_id'],
                        'speed': speed_est['speed'],
                        'timestamp': speed_est['timestamp']
                    })
        
        # Check for unusual detection patterns
        if len(self.detection_history) > 10:
            recent_counts = [entry['count'] for entry in list(self.detection_history)[-10:]]
            mean_count = np.mean(recent_counts)
            
            if len(self.detection_history) > 0:
                latest_count = self.detection_history[-1]['count']
                if latest_count > mean_count * 2:
                    anomalies.append({
                        'type': 'traffic_surge',
                        'count': latest_count,
                        'timestamp': self.detection_history[-1]['timestamp']
                    })
        
        return anomalies
