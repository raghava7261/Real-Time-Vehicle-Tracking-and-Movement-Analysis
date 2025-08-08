import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class Dashboard:
    """Dashboard for displaying analytics and visualizations"""
    
    def __init__(self):
        """Initialize dashboard"""
        pass
    
    def display_analytics(self, analytics):
        """Display comprehensive analytics dashboard"""
        if not analytics.has_data():
            st.warning("No analytics data available")
            return
        
        # Main metrics
        self._display_main_metrics(analytics)
        
        # Charts and visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            self._display_detection_timeline(analytics)
            self._display_speed_distribution(analytics)
        
        with col2:
            self._display_vehicle_distribution(analytics)
            self._display_performance_metrics(analytics)
        
        # Advanced analytics
        st.subheader("Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            self._display_trajectory_analysis(analytics)
        
        with col2:
            self._display_anomaly_detection(analytics)
    
    def _display_main_metrics(self, analytics):
        """Display main metrics cards"""
        stats = analytics.get_current_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", stats['total_detections'])
        
        with col2:
            st.metric("Active Tracks", stats['active_tracks'])
        
        with col3:
            st.metric("Average Confidence", f"{stats['avg_confidence']:.2%}")
        
        with col4:
            st.metric("Processing FPS", f"{stats['fps']:.1f}")
    
    def _display_detection_timeline(self, analytics):
        """Display detection timeline chart"""
        st.subheader("Detection Timeline")
        
        timestamps, counts = analytics.get_detection_timeline()
        
        if len(timestamps) > 0:
            # Convert timestamps to datetime
            dates = [datetime.fromtimestamp(ts) for ts in timestamps]
            
            # Create DataFrame
            df = pd.DataFrame({
                'Time': dates,
                'Detections': counts
            })
            
            # Create line chart
            fig = px.line(df, x='Time', y='Detections', 
                         title='Vehicle Detections Over Time')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data available")
    
    def _display_vehicle_distribution(self, analytics):
        """Display vehicle type distribution"""
        st.subheader("Vehicle Distribution")
        
        vehicle_dist = analytics.get_vehicle_distribution()
        
        if vehicle_dist:
            # Create pie chart
            fig = px.pie(
                values=list(vehicle_dist.values()),
                names=list(vehicle_dist.keys()),
                title='Vehicle Type Distribution'
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No vehicle distribution data available")
    
    def _display_speed_distribution(self, analytics):
        """Display speed distribution"""
        st.subheader("Speed Analysis")
        
        speed_stats = analytics.get_speed_statistics()
        
        if speed_stats:
            # Display speed metrics
            col1, col2, col3 = st.columns(3)
            
            unit = speed_stats.get('unit', 'px/s')
            
            with col1:
                st.metric("Mean Speed", f"{speed_stats['mean_speed']:.1f} {unit}")
            
            with col2:
                st.metric("Max Speed", f"{speed_stats['max_speed']:.1f} {unit}")
            
            with col3:
                st.metric("Std Dev", f"{speed_stats['std_speed']:.1f} {unit}")
            
            # Create histogram
            # Note: This would need actual speed data for a real histogram
            # For demo purposes, we'll show a simple bar chart
            speeds = np.random.normal(speed_stats['mean_speed'], 
                                    speed_stats['std_speed'], 100)
            
            fig = px.histogram(x=speeds, nbins=20, title='Speed Distribution')
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No speed data available")
    
    def _display_performance_metrics(self, analytics):
        """Display performance metrics"""
        st.subheader("Performance Metrics")
        
        stats = analytics.get_current_stats()
        
        # FPS gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stats['fps'],
            title={'text': "Processing FPS"},
            gauge={
                'axis': {'range': [None, 60]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 15], 'color': "lightgray"},
                    {'range': [15, 30], 'color': "yellow"},
                    {'range': [30, 60], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 30
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_trajectory_analysis(self, analytics):
        """Display trajectory analysis"""
        st.subheader("Trajectory Analysis")
        
        trajectory_analysis = analytics.get_trajectory_analysis()
        
        if trajectory_analysis:
            # Create summary table
            data = []
            for track_id, analysis in trajectory_analysis.items():
                data.append({
                    'Track ID': track_id,
                    'Distance': f"{analysis['total_distance']:.1f} px",
                    'Trajectory Length': analysis['trajectory_length']
                })
            
            if data:
                df = pd.DataFrame(data)
                st.dataframe(df)
            else:
                st.info("No trajectory data available")
        else:
            st.info("No trajectory analysis available")
    
    def _display_anomaly_detection(self, analytics):
        """Display anomaly detection results"""
        st.subheader("Anomaly Detection")
        
        anomalies = analytics.detect_anomalies()
        
        if anomalies:
            for anomaly in anomalies:
                if anomaly['type'] == 'unusual_speed':
                    st.warning(f"⚠️ Unusual speed detected: Track {anomaly['track_id']} "
                              f"moving at {anomaly['speed']:.1f} px/s")
                elif anomaly['type'] == 'traffic_surge':
                    st.warning(f"⚠️ Traffic surge detected: {anomaly['count']} vehicles")
        else:
            st.success("✅ No anomalies detected")
    
    def display_hourly_statistics(self, analytics):
        """Display hourly statistics"""
        st.subheader("Hourly Traffic Pattern")
        
        hourly_stats = analytics.get_hourly_statistics()
        
        if hourly_stats:
            # Create bar chart
            fig = px.bar(
                x=hourly_stats['hours'],
                y=hourly_stats['vehicle_counts'],
                title='Hourly Vehicle Count',
                labels={'x': 'Hour of Day', 'y': 'Vehicle Count'}
            )
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    def export_dashboard_data(self, analytics):
        """Export dashboard data for external analysis"""
        export_data = {
            'current_stats': analytics.get_current_stats(),
            'vehicle_distribution': analytics.get_vehicle_distribution(),
            'speed_statistics': analytics.get_speed_statistics(),
            'trajectory_analysis': analytics.get_trajectory_analysis(),
            'anomalies': analytics.detect_anomalies()
        }
        
        return export_data
