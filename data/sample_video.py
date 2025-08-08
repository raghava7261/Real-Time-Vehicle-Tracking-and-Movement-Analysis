# This file is reserved for sample video utilities
# Place your sample video file as 'data/sample_video.mp4' for testing

def get_sample_video_info():
    """Get information about sample video requirements"""
    return {
        'description': 'Sample video for vehicle tracking demonstration',
        'requirements': [
            'Video file should be placed at data/sample_video.mp4',
            'Supported formats: MP4, AVI, MOV, MKV',
            'Recommended resolution: 640x480 or higher',
            'Should contain moving vehicles for best results'
        ],
        'use_cases': [
            'Testing vehicle detection algorithms',
            'Demonstrating tracking capabilities',
            'Performance benchmarking',
            'Algorithm development'
        ]
    }
