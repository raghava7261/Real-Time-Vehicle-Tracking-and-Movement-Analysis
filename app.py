import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from utils.video_processor import VideoProcessor
from utils.dashboard import Dashboard
from utils.analytics import Analytics
import threading
import time

# Set page config
st.set_page_config(
    page_title="Real-Time Vehicle Tracking System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'analytics' not in st.session_state:
    st.session_state.analytics = Analytics()
if 'dashboard' not in st.session_state:
    st.session_state.dashboard = Dashboard()

def main():
    st.title("Real-Time Vehicle Tracking & Movement Analysis")
    st.markdown("---")
    
    # Sidebar for configuration
    st.sidebar.header("Configuration")
    
    # Video source selection
    video_source = st.sidebar.selectbox(
        "Select Video Source",
        ["Upload Video File", "Webcam", "Sample Video"]
    )
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.1, 1.0, 0.45, 0.05)
    
    # Tracking parameters
    st.sidebar.subheader("Tracking Parameters")
    max_disappeared = st.sidebar.slider("Max Disappeared Frames", 5, 50, 20, 5)
    max_distance = st.sidebar.slider("Max Distance for Tracking", 50, 200, 100, 10)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Video Processing")
        
        if video_source == "Upload Video File":
            uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    video_path = tmp_file.name
                
                if st.button("Start Processing"):
                    process_video(video_path, confidence_threshold, iou_threshold, max_disappeared, max_distance)
        
        elif video_source == "Webcam":
            if st.button("Start Webcam Processing"):
                process_webcam(confidence_threshold, iou_threshold, max_disappeared, max_distance)
        
        elif video_source == "Sample Video":
            st.info("Sample video processing will use your sample video file.")
            if st.button("Start Sample Processing"):
                process_sample_video(confidence_threshold, iou_threshold, max_disappeared, max_distance)
    
    with col2:
        st.subheader("Live Statistics")
        display_live_stats()
    
    # Analytics section
    st.markdown("---")
    st.subheader("Analytics Dashboard")
    
    # Display analytics if available
    if st.session_state.analytics.has_data():
        st.session_state.dashboard.display_analytics(st.session_state.analytics)
    else:
        st.info("No analytics data available. Start video processing to see analytics.")
    
    # Controls
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Reset Analytics"):
            st.session_state.analytics.reset()
            st.success("Analytics reset successfully!")
    
    with col2:
        if st.button("Export Report"):
            export_report()
    
    with col3:
        if st.button("Stop Processing"):
            stop_processing()

def process_video(video_path, confidence_threshold, iou_threshold, max_disappeared, max_distance):
    """Process uploaded video file"""
    try:
        st.session_state.video_processor = VideoProcessor(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_disappeared=max_disappeared,
            max_distance=max_distance
        )
        
        st.session_state.processing = True
        
        # Create placeholders for video display
        video_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        # Process video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while cap.isOpened() and st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, detections, tracks = st.session_state.video_processor.process_frame(frame)
            
            # Update analytics
            st.session_state.analytics.update(detections, tracks)
            
            # Display frame
            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            
            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            # Small delay to prevent overwhelming
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
        st.session_state.processing = False
        st.success("Video processing completed!")
        
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        st.session_state.processing = False

def process_webcam(confidence_threshold, iou_threshold, max_disappeared, max_distance):
    """Process webcam feed"""
    try:
        st.session_state.video_processor = VideoProcessor(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            max_disappeared=max_disappeared,
            max_distance=max_distance
        )
        
        st.session_state.processing = True
        
        # Create placeholders
        video_placeholder = st.empty()
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Cannot open webcam. Please check your camera connection.")
            return
        
        while st.session_state.processing:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from webcam")
                break
            
            # Process frame
            processed_frame, detections, tracks = st.session_state.video_processor.process_frame(frame)
            
            # Update analytics
            st.session_state.analytics.update(detections, tracks)
            
            # Display frame
            video_placeholder.image(processed_frame, channels="BGR", use_container_width=True)
            
            # Small delay
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
        st.session_state.processing = False
        
    except Exception as e:
        st.error(f"Error processing webcam: {str(e)}")
        st.session_state.processing = False

def process_sample_video(confidence_threshold, iou_threshold, max_disappeared, max_distance):
    """Process sample video for demonstration"""
    # Check if sample video exists
    sample_video_path = "data/sample_video.mp4"
    
    if os.path.exists(sample_video_path):
        st.info("Processing sample video...")
        process_video(sample_video_path, confidence_threshold, iou_threshold, max_disappeared, max_distance)
    else:
        st.error(f"Sample video not found at {sample_video_path}")
        st.info("Please place your sample video file at 'data/sample_video.mp4' or use the upload feature.")

def display_live_stats():
    """Display live statistics"""
    if st.session_state.analytics.has_data():
        stats = st.session_state.analytics.get_current_stats()
        
        st.metric("Total Vehicles Detected", stats['total_detections'])
        st.metric("Active Tracks", stats['active_tracks'])
        st.metric("Average Confidence", f"{stats['avg_confidence']:.2f}")
        st.metric("Processing FPS", f"{stats['fps']:.1f}")
        
        # Vehicle type distribution
        st.subheader("Vehicle Types")
        for vehicle_type, count in stats['vehicle_types'].items():
            st.write(f"{vehicle_type.capitalize()}: {count}")
    else:
        st.info("No live data available")

def export_report():
    """Export analytics report"""
    if st.session_state.analytics.has_data():
        report = st.session_state.analytics.generate_report()
        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"vehicle_tracking_report_{int(time.time())}.json",
            mime="application/json"
        )
    else:
        st.warning("No data available for export")

def stop_processing():
    """Stop video processing"""
    st.session_state.processing = False
    st.info("Processing stopped")

if __name__ == "__main__":
    main()
