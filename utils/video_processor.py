import cv2
import numpy as np
import time
import logging
from utils.vehicle_detector import VehicleDetector
from utils.tracker import MultiObjectTracker

class VideoProcessor:
    """Main video processing pipeline"""
    
    def __init__(self, confidence_threshold=0.5, iou_threshold=0.45, 
                 max_disappeared=20, max_distance=100):
        """
        Initialize video processor
        
        Args:
            confidence_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            max_disappeared: Max frames for track disappearance
            max_distance: Max distance for track association
        """
        self.detector = VehicleDetector(
            confidence_threshold=confidence_threshold
        )
        
        self.tracker = MultiObjectTracker(
            max_disappeared=max_disappeared,
            max_distance=max_distance
        )
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        logging.info("Video processor initialized")
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            Tuple of (processed_frame, detections, tracks)
        """
        start_time = time.time()
        
        # Detect vehicles
        detections = self.detector.detect(frame)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Draw results on frame
        output_frame = self._draw_results(frame, detections, tracks)
        
        # Update performance metrics
        self.frame_count += 1
        processing_time = time.time() - start_time
        
        if processing_time > 0:
            current_fps = 1.0 / processing_time
            self.fps = 0.9 * self.fps + 0.1 * current_fps  # Smooth FPS
        
        return output_frame, detections, tracks
    
    def _draw_results(self, frame, detections, tracks):
        """Draw detection and tracking results on frame"""
        output_frame = frame.copy()
        
        # Draw detections
        output_frame = self.detector.draw_detections(output_frame, detections)
        
        # Draw tracks
        output_frame = self.tracker.draw_tracks(output_frame, tracks)
        
        # Draw performance info
        self._draw_performance_info(output_frame)
        
        # Draw statistics
        self._draw_statistics(output_frame, detections, tracks)
        
        return output_frame
    
    def _draw_performance_info(self, frame):
        """Draw performance information on frame"""
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Processing time
        elapsed_time = time.time() - self.start_time
        cv2.putText(frame, f"Time: {elapsed_time:.1f}s", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _draw_statistics(self, frame, detections, tracks):
        """Draw current statistics on frame"""
        height, width = frame.shape[:2]
        
        # Detection count
        cv2.putText(frame, f"Detections: {len(detections)}", 
                   (width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Active tracks
        cv2.putText(frame, f"Active Tracks: {len(tracks)}", 
                   (width - 200, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Vehicle type counts
        vehicle_counts = {}
        for detection in detections:
            vehicle_type = detection['class']
            vehicle_counts[vehicle_type] = vehicle_counts.get(vehicle_type, 0) + 1
        
        y_offset = 90
        for vehicle_type, count in vehicle_counts.items():
            cv2.putText(frame, f"{vehicle_type.capitalize()}: {count}", 
                       (width - 200, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            y_offset += 25
    
    def process_video_file(self, video_path, output_path=None):
        """
        Process entire video file
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (optional)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logging.info(f"Processing video: {video_path}")
        logging.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, detections, tracks = self.process_frame(frame)
            
            # Write frame if output specified
            if writer:
                writer.write(processed_frame)
            
            frame_count += 1
            
            # Progress update
            if frame_count % 100 == 0:
                progress = (frame_count / total_frames) * 100
                logging.info(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        logging.info(f"Video processing completed: {frame_count} frames processed")
    
    def update_parameters(self, confidence_threshold=None, max_disappeared=None, 
                         max_distance=None):
        """Update processing parameters"""
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
            self.detector.update_confidence_threshold(confidence_threshold)
        
        if max_disappeared is not None:
            self.tracker.max_disappeared = max_disappeared
        
        if max_distance is not None:
            self.tracker.max_distance = max_distance
        
        logging.info("Processing parameters updated")
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'processing_time': time.time() - self.start_time,
            'avg_fps': self.frame_count / (time.time() - self.start_time) if self.frame_count > 0 else 0
        }
