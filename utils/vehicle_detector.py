import cv2
import numpy as np
import logging

# Alternative implementation without ultralytics/YOLO
# Using OpenCV's built-in background subtraction for vehicle detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class VehicleDetector:
    """Vehicle detection using background subtraction (OpenCV)"""
    
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize vehicle detector
        
        Args:
            model_path: Path to YOLO model (ignored if YOLO not available)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                self.use_yolo = True
                # Vehicle classes in COCO dataset
                self.vehicle_classes = {
                    2: 'car',
                    3: 'motorcycle', 
                    5: 'bus',
                    7: 'truck'
                }
                logging.info(f"Vehicle detector initialized with YOLO model: {model_path}")
            except Exception as e:
                logging.warning(f"YOLO initialization failed: {e}. Using background subtraction.")
                self.use_yolo = False
                self._init_background_subtraction()
        else:
            logging.info("YOLO not available. Using background subtraction for vehicle detection.")
            self.use_yolo = False
            self._init_background_subtraction()
    
    def _init_background_subtraction(self):
        """Initialize background subtraction method"""
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50,
            history=200
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_contour_area = 500
        self.vehicle_classes = ['car', 'truck', 'bus', 'motorcycle']
    
    def detect(self, frame):
        """
        Detect vehicles in frame
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            List of detections with format:
            [{'class': str, 'confidence': float, 'bbox': [x, y, w, h]}]
        """
        try:
            if self.use_yolo:
                return self._detect_with_yolo(frame)
            else:
                return self._detect_with_background_subtraction(frame)
            
        except Exception as e:
            logging.error(f"Error during detection: {e}")
            return []
    
    def _detect_with_yolo(self, frame):
        """Detect vehicles using YOLO"""
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID
                    class_id = int(box.cls.cpu().numpy()[0])
                    
                    # Check if it's a vehicle class
                    if class_id in self.vehicle_classes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0]
                        
                        # Convert to x, y, w, h format
                        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                        
                        # Get confidence
                        confidence = float(box.conf.cpu().numpy()[0])
                        
                        detection = {
                            'class': self.vehicle_classes[class_id],
                            'confidence': confidence,
                            'bbox': [x, y, w, h],
                            'center': [x + w//2, y + h//2]
                        }
                        
                        detections.append(detection)
        
        return detections
    
    def _detect_with_background_subtraction(self, frame):
        """Detect vehicles using background subtraction"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0
        
        # Morphological operations to clean up the mask
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            # Filter by area
            if cv2.contourArea(contour) > self.min_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (typical for vehicles)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 5.0:
                    # Estimate confidence based on contour area
                    area = cv2.contourArea(contour)
                    confidence = min(0.9, area / 10000)  # Normalize to 0-0.9
                    
                    if confidence > self.confidence_threshold:
                        # Randomly assign vehicle class for demo
                        import random
                        vehicle_class = random.choice(self.vehicle_classes)
                        
                        detection = {
                            'class': vehicle_class,
                            'confidence': confidence,
                            'bbox': [x, y, w, h],
                            'center': [x + w//2, y + h//2]
                        }
                        
                        detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        output_frame = frame.copy()
        
        # Color mapping for different vehicle types
        colors = {
            'car': (0, 255, 0),      # Green
            'truck': (255, 0, 0),    # Blue
            'bus': (0, 0, 255),      # Red
            'motorcycle': (255, 255, 0)  # Cyan
        }
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            vehicle_class = detection['class']
            confidence = detection['confidence']
            
            # Get color for vehicle type
            color = colors.get(vehicle_class, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{vehicle_class}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(output_frame, (x, y - label_size[1] - 10), 
                         (x + label_size[0], y), color, -1)
            
            # Draw label text
            cv2.putText(output_frame, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return output_frame
    
    def update_confidence_threshold(self, threshold):
        """Update confidence threshold"""
        self.confidence_threshold = threshold
        logging.info(f"Confidence threshold updated to: {threshold}")
