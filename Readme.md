# Real-Time Vehicle Tracking and Movement Analysis System

A comprehensive real-time vehicle tracking and movement analysis system built with Streamlit and computer vision techniques. This system uses advanced detection algorithms, Kalman filters for tracking, and provides a dashboard for analytics and visualization.

## Features

### 🚗 Vehicle Detection
- **Adaptive Detection**: Uses YOLO when available, falls back to background subtraction
- **Multi-Class Support**: Detects cars, trucks, buses, and motorcycles
- **Real-time Processing**: Optimized for live video streams
- **Configurable Parameters**: Adjustable confidence thresholds and detection settings

### 📊 Multi-Object Tracking
- **Kalman Filter Integration**: Smooth trajectory prediction and tracking
- **Identity Maintenance**: Consistent vehicle tracking across frames
- **Occlusion Handling**: Robust tracking through temporary obstructions
- **Trajectory Analysis**: Complete path tracking and movement analysis

### 📈 Analytics Dashboard
- **Real-time Statistics**: Live vehicle counts, speeds, and classifications
- **Interactive Visualizations**: Plotly-based charts and graphs
- **Performance Metrics**: FPS monitoring and processing statistics
- **Export Capabilities**: Download analytics reports in JSON format

### 🎥 Video Processing
- **Multiple Input Sources**: Support for uploaded videos, webcam, and sample videos
- **Live Processing**: Real-time video analysis and visualization
- **Batch Processing**: Complete video file analysis with progress tracking
- **Sample Video Generator**: Built-in traffic simulation for testing

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenCV compatible system
- Optional: CUDA-capable GPU for enhanced performance

### Dependencies
```bash
pip install streamlit opencv-python numpy pandas plotly scipy pillow
```

Optional (for enhanced YOLO detection):
```bash
pip install ultralytics
```

## Usage

### Starting the Application
```bash
streamlit run app.py --server.port 5000
```

### Using the Interface

1. **Configuration Panel**: Adjust detection and tracking parameters in the sidebar
2. **Video Source Selection**: Choose between uploaded video, webcam, or sample video
3. **Processing Controls**: Start/stop video processing with real-time feedback
4. **Analytics Dashboard**: View comprehensive statistics and visualizations

### Video Sources

#### Upload Video File
- Supported formats: MP4, AVI, MOV, MKV
- Drag and drop or browse to select files
- Real-time processing with progress tracking

#### Webcam Processing
- Live camera feed processing
- Automatic device detection
- Real-time analytics updates

#### Sample Video
- Built-in traffic simulation
- Perfect for testing and demonstration
- Configurable vehicle patterns

## System Architecture

### Core Components

1. **Video Processor** (`utils/video_processor.py`)
   - Main processing pipeline
   - Coordinates detection and tracking
   - Performance monitoring

2. **Vehicle Detector** (`utils/vehicle_detector.py`)
   - YOLO-based detection (when available)
   - Background subtraction fallback
   - Multi-class vehicle classification

3. **Multi-Object Tracker** (`utils/tracker.py`)
   - Kalman filter implementation
   - Hungarian algorithm for association
   - Track lifecycle management

4. **Analytics Engine** (`utils/analytics.py`)
   - Real-time statistics calculation
   - Speed estimation algorithms
   - Anomaly detection

5. **Dashboard** (`utils/dashboard.py`)
   - Interactive visualizations
   - Real-time data presentation
   - Export functionality

### Data Flow

```
Video Input → Frame Processing → Vehicle Detection → Multi-Object Tracking → Analytics → Visualization
```

## Configuration

### Detection Parameters
- **Confidence Threshold**: Minimum confidence for valid detections (0.1-1.0)
- **IoU Threshold**: Intersection over Union threshold for non-max suppression
- **Processing Resolution**: Video resolution for optimal performance

### Tracking Parameters
- **Max Disappeared Frames**: Maximum frames a track can be missing (5-50)
- **Max Distance**: Maximum distance for track association (50-200 pixels)
- **Trajectory Length**: Number of historical positions to maintain

## Performance Optimization

### System Requirements
- **CPU**: Dual-core minimum, quad-core recommended
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional CUDA support for enhanced performance
- **Storage**: Minimal requirements (no persistent storage needed)

### Optimization Tips
1. Adjust video resolution for performance vs. accuracy balance
2. Use GPU acceleration when available
3. Configure detection parameters based on video characteristics
4. Monitor system resources during processing

## Analytics and Reporting

### Available Metrics
- **Vehicle Counts**: Total detections and active tracks
- **Speed Analysis**: Average, maximum, and distribution statistics
- **Vehicle Classification**: Distribution by vehicle type
- **Trajectory Analysis**: Path lengths and movement patterns
- **Performance Metrics**: Processing FPS and system efficiency

### Export Options
- **JSON Reports**: Comprehensive analytics data
- **Real-time Data**: Live streaming of statistics
- **Historical Analysis**: Trend analysis over time

## Troubleshooting

### Common Issues

1. **No Video Display**
   - Check video file format compatibility
   - Verify camera permissions for webcam
   - Ensure OpenCV installation is complete

2. **Low Performance**
   - Reduce video resolution
   - Adjust confidence thresholds
   - Check system resources

3. **Inaccurate Detection**
   - Adjust detection parameters
   - Ensure proper lighting conditions
   - Consider video quality and angle

### Error Messages
- **"No module named 'cv2'"**: Install OpenCV with `pip install opencv-python`
- **"Cannot open webcam"**: Check camera permissions and availability
- **"YOLO not available"**: Install ultralytics or use background subtraction mode

## Development

### Project Structure
```
├── app.py                 # Main Streamlit application
├── utils/
│   ├── video_processor.py # Video processing pipeline
│   ├── vehicle_detector.py# Vehicle detection algorithms
│   ├── tracker.py         # Multi-object tracking
│   ├── analytics.py       # Analytics and reporting
│   └── dashboard.py       # Visualization dashboard
├── models/
│   └── kalman_filter.py   # Kalman filter implementation
├── data/
│   └── sample_video.py    # Sample video generation
└── .streamlit/
    └── config.toml        # Streamlit configuration
```

### Extending the System

1. **Custom Detection Models**: Replace YOLO with custom trained models
2. **Advanced Analytics**: Add custom metrics and analysis algorithms
3. **Database Integration**: Store historical data for long-term analysis
4. **API Integration**: Connect to external traffic management systems

## Applications

### Traffic Management
- Intersection monitoring and optimization
- Traffic flow analysis and prediction
- Automated traffic light control
- Emergency vehicle detection and priority routing

### Smart City Integration
- Integration with Intelligent Transportation Systems (ITS)
- Real-time traffic data for navigation applications
- Environmental impact assessment
- Public transportation optimization

### Security and Law Enforcement
- Automated license plate recognition (with additional modules)
- Speed violation detection and enforcement
- Parking management and violation detection
- Vehicle theft prevention and recovery

### Business Intelligence
- Retail location analysis based on traffic patterns
- Drive-through optimization for restaurants and banks
- Parking facility management and optimization
- Fleet tracking and management services

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests, report issues, or suggest improvements.

## Support

For support and questions, please refer to the project documentation or create an issue in the repository.

---

**Note**: This system is designed for educational and research purposes. For production deployment, additional considerations for scalability, security, and regulatory compliance may be required.