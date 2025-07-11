"""
Multi-Model Streamlit Interface for Autonomous Driving Perception
Supports ensemble predictions, domain adaptation, and model comparison
"""

import streamlit as st
import requests
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import time
from pathlib import Path
import yaml
from typing import Dict, List, Optional
import logging
from datetime import datetime
import io
import base64


# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Model Autonomous Driving Perception",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Load configuration
@st.cache_resource
def load_config():
    """Load configuration file"""
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()


def main():
    """Main Streamlit application"""
    
    st.title("🚗 Multi-Model Autonomous Driving Perception")
    st.markdown("---")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        [
            "🏠 Home",
            "🔍 Single Image Detection",
            "🎯 Ensemble Predictions",
            "🔄 Domain Adaptation",
            "🎥 Video Processing",
            "📊 Model Comparison",
            "⚡ Performance Benchmarking",
            "🛠️ Model Management",
            "📈 Training Dashboard"
        ]
    )
    
    # Route to selected page
    if page == "🏠 Home":
        home_page()
    elif page == "🔍 Single Image Detection":
        single_image_detection_page()
    elif page == "🎯 Ensemble Predictions":
        ensemble_predictions_page()
    elif page == "🔄 Domain Adaptation":
        domain_adaptation_page()
    elif page == "🎥 Video Processing":
        video_processing_page()
    elif page == "📊 Model Comparison":
        model_comparison_page()
    elif page == "⚡ Performance Benchmarking":
        performance_benchmarking_page()
    elif page == "🛠️ Model Management":
        model_management_page()
    elif page == "📈 Training Dashboard":
        training_dashboard_page()


def home_page():
    """Home page with system overview"""
    
    st.header("🏠 System Overview")
    
    # Check API health
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            st.success("✅ API is healthy and running")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("API Version", health_data.get('version', 'N/A'))
            
            with col2:
                st.metric("Models Loaded", health_data.get('models_loaded', 0))
            
            with col3:
                st.metric("Domain Adaptation", "Enabled" if health_data.get('domain_adaptation_enabled') else "Disabled")
            
            with col4:
                st.metric("Status", health_data.get('status', 'Unknown'))
        else:
            st.error("❌ API is not responding")
    
    except Exception as e:
        st.error(f"❌ Cannot connect to API: {str(e)}")
    
    # System capabilities
    st.subheader("🎯 System Capabilities")
    
    capabilities = [
        "🔍 **Multi-Model Object Detection**: YOLOv8, RetinaNet, EfficientDet",
        "🎯 **Ensemble Predictions**: Weighted averaging, max voting, confidence weighting",
        "🔄 **Domain Adaptation**: DANN, CORAL, MMD for CARLA→KITTI adaptation",
        "🎥 **Video Processing**: Real-time video analysis with tracking",
        "📊 **Model Comparison**: Side-by-side performance analysis",
        "⚡ **Performance Benchmarking**: Speed and accuracy measurements",
        "🛠️ **Model Management**: Dynamic weight adjustment and configuration"
    ]
    
    for capability in capabilities:
        st.markdown(capability)
    
    # Project information
    st.subheader("📋 Project Information")
    
    project_info = config.get('project', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Project**: {project_info.get('name', 'N/A')}
        **Version**: {project_info.get('version', 'N/A')}
        **Description**: {project_info.get('description', 'N/A')}
        """)
    
    with col2:
        st.info(f"""
        **Supported Datasets**: CARLA, KITTI, nuScenes
        **Domain Adaptation**: Simulation → Real World
        **Model Types**: Detection, Segmentation, Tracking, Depth
        """)


def single_image_detection_page():
    """Single image detection with model selection"""
    
    st.header("🔍 Single Image Detection")
    
    # Model selection
    st.subheader("🎛️ Model Selection")
    
    # Get available models
    try:
        response = requests.get(f"{API_BASE_URL}/models/info", timeout=5)
        if response.status_code == 200:
            models_info = response.json()
            available_models = list(models_info['models'].keys())
        else:
            available_models = ['yolov8', 'retinanet', 'efficientdet']
    except:
        available_models = ['yolov8', 'retinanet', 'efficientdet']
    
    selected_model = st.selectbox("Select Model", available_models)
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    
    with col2:
        use_domain_adaptation = st.checkbox("Use Domain Adaptation", value=True)
    
    # Image upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Display uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        
        # Prediction button
        if st.button("🔍 Run Detection"):
            with st.spinner("Running detection..."):
                try:
                    # Prepare file for API
                    uploaded_file.seek(0)
                    files = {"file": uploaded_file}
                    params = {
                        "confidence_threshold": confidence_threshold
                    }
                    
                    # Make API request
                    response = requests.post(
                        f"{API_BASE_URL}/predict/model/{selected_model}",
                        files=files,
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.success("✅ Detection completed!")
                        
                        # Show prediction details
                        st.subheader("📊 Detection Results")
                        
                        prediction = result['prediction']
                        
                        if prediction['type'] == 'detection':
                            predictions_data = prediction['predictions']
                            
                            if len(predictions_data) > 0 and len(predictions_data[0]['boxes']) > 0:
                                # Create results dataframe
                                results_df = pd.DataFrame({
                                    'Class': predictions_data[0]['classes'],
                                    'Confidence': predictions_data[0]['scores'],
                                    'Bounding Box': [f"({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})" 
                                                   for box in predictions_data[0]['boxes']]
                                })
                                
                                st.dataframe(results_df)
                                
                                # Visualize detections
                                visualized_image = visualize_detections(image, predictions_data[0])
                                st.image(cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB), 
                                        caption="Detection Results", use_column_width=True)
                            else:
                                st.info("No objects detected")
                        
                        elif prediction['type'] == 'classification':
                            st.write(f"**Predicted Class**: {prediction['predictions'][0]}")
                            st.write(f"**Confidence**: {prediction['confidence_scores'][0]:.3f}")
                            
                            # Show class probabilities
                            if 'probabilities' in prediction:
                                prob_df = pd.DataFrame({
                                    'Class': range(len(prediction['probabilities'][0])),
                                    'Probability': prediction['probabilities'][0]
                                })
                                
                                fig = px.bar(prob_df, x='Class', y='Probability', 
                                           title='Class Probabilities')
                                st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def ensemble_predictions_page():
    """Ensemble predictions with multiple models"""
    
    st.header("🎯 Ensemble Predictions")
    
    # Ensemble configuration
    st.subheader("⚙️ Ensemble Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    
    with col2:
        use_domain_adaptation = st.checkbox("Use Domain Adaptation", value=True)
    
    with col3:
        return_individual = st.checkbox("Show Individual Predictions", value=False)
    
    # Image upload
    st.subheader("📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], key="ensemble_upload")
    
    if uploaded_file is not None:
        # Display uploaded image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
        
        # Prediction button
        if st.button("🎯 Run Ensemble Prediction"):
            with st.spinner("Running ensemble prediction..."):
                try:
                    # Prepare file for API
                    uploaded_file.seek(0)
                    files = {"file": uploaded_file}
                    params = {
                        "use_domain_adaptation": use_domain_adaptation,
                        "confidence_threshold": confidence_threshold,
                        "return_individual_predictions": return_individual
                    }
                    
                    # Make API request
                    response = requests.post(
                        f"{API_BASE_URL}/predict/ensemble",
                        files=files,
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results
                        st.success("✅ Ensemble prediction completed!")
                        
                        # Show processing info
                        processing_info = result.get('processing_info', {})
                        st.info(f"""
                        **Ensemble Method**: {processing_info.get('ensemble_method', 'N/A')}
                        **Domain Adaptation**: {processing_info.get('domain_adaptation_used', False)}
                        **Models Used**: {', '.join(processing_info.get('models_used', []))}
                        """)
                        
                        # Show ensemble results
                        predictions = result['predictions']
                        
                        if 'ensemble' in predictions:
                            st.subheader("🎯 Ensemble Results")
                            
                            ensemble_pred = predictions['ensemble']
                            
                            if 'detection' in ensemble_pred:
                                detection_pred = ensemble_pred['detection']
                                
                                if len(detection_pred['boxes']) > 0:
                                    # Create results dataframe
                                    results_df = pd.DataFrame({
                                        'Class': detection_pred['classes'],
                                        'Confidence': detection_pred['scores'],
                                        'Model Source': detection_pred.get('model_sources', ['N/A'] * len(detection_pred['classes'])),
                                        'Bounding Box': [f"({box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f})" 
                                                       for box in detection_pred['boxes']]
                                    })
                                    
                                    st.dataframe(results_df)
                                    
                                    # Visualize ensemble detections
                                    visualized_image = visualize_detections(image, detection_pred)
                                    st.image(cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB), 
                                            caption="Ensemble Detection Results", use_column_width=True)
                                else:
                                    st.info("No objects detected by ensemble")
                        
                        # Show individual model results if requested
                        if return_individual and 'individual' in predictions:
                            st.subheader("🔍 Individual Model Results")
                            
                            individual_preds = predictions['individual']
                            
                            tabs = st.tabs(list(individual_preds.keys()))
                            
                            for tab, (model_name, pred) in zip(tabs, individual_preds.items()):
                                with tab:
                                    st.write(f"**Model**: {model_name}")
                                    
                                    if pred['type'] == 'detection':
                                        pred_data = pred['predictions'][0]
                                        
                                        if len(pred_data['boxes']) > 0:
                                            model_df = pd.DataFrame({
                                                'Class': pred_data['classes'],
                                                'Confidence': pred_data['scores']
                                            })
                                            st.dataframe(model_df)
                                        else:
                                            st.info("No detections")
                    
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def domain_adaptation_page():
    """Domain adaptation training and evaluation"""
    
    st.header("🔄 Domain Adaptation")
    
    # Training section
    st.subheader("🏋️ Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_dataset = st.selectbox("Source Dataset", ["carla", "airsim"])
        target_dataset = st.selectbox("Target Dataset", ["kitti", "nuscenes"])
    
    with col2:
        num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=200, value=50)
        adaptation_methods = st.multiselect(
            "Adaptation Methods",
            ["dann", "coral", "mmd"],
            default=["dann", "coral"]
        )
    
    # Start training
    if st.button("🚀 Start Domain Adaptation Training"):
        if adaptation_methods:
            with st.spinner("Starting training..."):
                try:
                    params = {
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "num_epochs": num_epochs,
                        "adaptation_methods": adaptation_methods
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/domain_adaptation/train",
                        params=params,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Training started! Training ID: {result['training_id']}")
                        
                        # Store training ID in session state
                        st.session_state['training_id'] = result['training_id']
                    else:
                        st.error(f"❌ Training failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("Please select at least one adaptation method")
    
    # Training status
    if 'training_id' in st.session_state:
        st.subheader("📊 Training Status")
        
        training_id = st.session_state['training_id']
        
        if st.button("🔄 Refresh Status"):
            try:
                response = requests.get(f"{API_BASE_URL}/domain_adaptation/status/{training_id}", timeout=5)
                
                if response.status_code == 200:
                    status = response.json()
                    
                    # Progress bar
                    st.progress(status.get('progress', 0))
                    
                    # Status metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Status", status.get('status', 'Unknown'))
                    
                    with col2:
                        st.metric("Current Epoch", f"{status.get('current_epoch', 0)}/{status.get('total_epochs', 0)}")
                    
                    with col3:
                        st.metric("Progress", f"{status.get('progress', 0):.1%}")
                    
                    # Loss visualization
                    if 'losses' in status:
                        losses = status['losses']
                        
                        loss_df = pd.DataFrame({
                            'Loss Type': list(losses.keys()),
                            'Value': list(losses.values())
                        })
                        
                        fig = px.bar(loss_df, x='Loss Type', y='Value', title='Training Losses')
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"❌ Status check failed: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def video_processing_page():
    """Video processing with ensemble models"""
    
    st.header("🎥 Video Processing")
    
    # Video upload
    st.subheader("📤 Upload Video")
    uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov', 'mkv'])
    
    if uploaded_video is not None:
        # Display video info
        st.video(uploaded_video)
        
        # Processing parameters
        st.subheader("⚙️ Processing Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            use_domain_adaptation = st.checkbox("Use Domain Adaptation", value=True, key="video_da")
        
        with col2:
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1, key="video_conf")
        
        with col3:
            enable_tracking = st.checkbox("Enable Tracking", value=True)
        
        # Upload video
        if st.button("📤 Upload Video for Processing"):
            with st.spinner("Uploading video..."):
                try:
                    files = {"file": uploaded_video}
                    params = {
                        "use_domain_adaptation": use_domain_adaptation,
                        "confidence_threshold": confidence_threshold,
                        "enable_tracking": enable_tracking
                    }
                    
                    response = requests.post(
                        f"{API_BASE_URL}/video/process_ensemble",
                        files=files,
                        params=params,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"✅ Video uploaded! Video ID: {result['video_id']}")
                        
                        # Store video ID in session state
                        st.session_state['video_id'] = result['video_id']
                    else:
                        st.error(f"❌ Upload failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Video processing status
    if 'video_id' in st.session_state:
        st.subheader("🎬 Video Processing")
        
        video_id = st.session_state['video_id']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Processing"):
                try:
                    response = requests.post(f"{API_BASE_URL}/video/process/{video_id}", timeout=10)
                    
                    if response.status_code == 200:
                        st.success("✅ Processing started!")
                    else:
                        st.error(f"❌ Processing failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🔄 Check Status"):
                try:
                    response = requests.get(f"{API_BASE_URL}/video/status/{video_id}", timeout=5)
                    
                    if response.status_code == 200:
                        status = response.json()
                        
                        # Progress bar
                        st.progress(status.get('progress', 0))
                        
                        # Status info
                        st.write(f"**Status**: {status.get('status', 'Unknown')}")
                        st.write(f"**Progress**: {status.get('processed_frames', 0)}/{status.get('total_frames', 0)} frames")
                        
                        # Download link if completed
                        if status.get('status') == 'completed':
                            st.success("✅ Processing completed!")
                            
                            download_url = f"{API_BASE_URL}/stream_processed_video/{video_id}"
                            st.markdown(f"[📥 Download Processed Video]({download_url})")
                    
                    else:
                        st.error(f"❌ Status check failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


def model_comparison_page():
    """Compare different models side by side"""
    
    st.header("📊 Model Comparison")
    
    # Model selection for comparison
    st.subheader("🎛️ Select Models to Compare")
    
    try:
        response = requests.get(f"{API_BASE_URL}/models/info", timeout=5)
        if response.status_code == 200:
            models_info = response.json()
            available_models = list(models_info['models'].keys())
        else:
            available_models = ['yolov8', 'retinanet', 'efficientdet']
    except:
        available_models = ['yolov8', 'retinanet', 'efficientdet']
    
    selected_models = st.multiselect("Select Models", available_models, default=available_models[:2])
    
    if len(selected_models) >= 2:
        # Image upload
        st.subheader("📤 Upload Test Image")
        uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'], key="comparison_upload")
        
        if uploaded_file is not None:
            # Display uploaded image
            image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Test Image", use_column_width=True)
            
            # Run comparison
            if st.button("🔍 Compare Models"):
                comparison_results = {}
                
                # Run predictions for each model
                for model_name in selected_models:
                    with st.spinner(f"Running {model_name}..."):
                        try:
                            uploaded_file.seek(0)
                            files = {"file": uploaded_file}
                            params = {"confidence_threshold": 0.5}
                            
                            response = requests.post(
                                f"{API_BASE_URL}/predict/model/{model_name}",
                                files=files,
                                params=params,
                                timeout=30
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                comparison_results[model_name] = result
                            else:
                                st.error(f"❌ {model_name} prediction failed")
                                
                        except Exception as e:
                            st.error(f"❌ Error with {model_name}: {str(e)}")
                
                # Display comparison results
                if comparison_results:
                    st.subheader("📊 Comparison Results")
                    
                    # Create comparison table
                    comparison_data = []
                    
                    for model_name, result in comparison_results.items():
                        prediction = result['prediction']
                        
                        if prediction['type'] == 'detection':
                            pred_data = prediction['predictions'][0]
                            num_detections = len(pred_data['boxes'])
                            avg_confidence = np.mean(pred_data['scores']) if pred_data['scores'] else 0
                        else:
                            num_detections = 1
                            avg_confidence = prediction.get('confidence_scores', [0])[0]
                        
                        comparison_data.append({
                            'Model': model_name,
                            'Detections': num_detections,
                            'Avg Confidence': f"{avg_confidence:.3f}",
                            'Type': prediction['type']
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df)
                    
                    # Side-by-side visualization
                    st.subheader("👁️ Visual Comparison")
                    
                    cols = st.columns(len(selected_models))
                    
                    for i, (model_name, result) in enumerate(comparison_results.items()):
                        with cols[i]:
                            st.write(f"**{model_name}**")
                            
                            prediction = result['prediction']
                            
                            if prediction['type'] == 'detection':
                                pred_data = prediction['predictions'][0]
                                visualized = visualize_detections(image, pred_data)
                                st.image(cv2.cvtColor(visualized, cv2.COLOR_BGR2RGB), 
                                        use_column_width=True)
                            else:
                                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                        use_column_width=True)
                                st.write(f"Class: {prediction['predictions'][0]}")
                                st.write(f"Confidence: {prediction['confidence_scores'][0]:.3f}")
    
    else:
        st.info("Please select at least 2 models for comparison")


def performance_benchmarking_page():
    """Performance benchmarking of models"""
    
    st.header("⚡ Performance Benchmarking")
    
    # Run benchmark
    if st.button("🚀 Run Benchmark"):
        with st.spinner("Running benchmark..."):
            try:
                response = requests.get(f"{API_BASE_URL}/benchmark/models", timeout=60)
                
                if response.status_code == 200:
                    benchmark_results = response.json()
                    
                    st.success("✅ Benchmark completed!")
                    
                    # Display results
                    st.subheader("📊 Benchmark Results")
                    
                    results_data = []
                    for model_name, metrics in benchmark_results['benchmark_results'].items():
                        results_data.append({
                            'Model': model_name,
                            'Avg Inference Time (s)': f"{metrics['avg_inference_time']:.4f}",
                            'FPS': f"{metrics['fps']:.2f}",
                            'Parameters': f"{metrics.get('model_parameters', 'N/A'):,}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df)
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Inference time comparison
                        inference_times = [metrics['avg_inference_time'] for metrics in benchmark_results['benchmark_results'].values()]
                        model_names = list(benchmark_results['benchmark_results'].keys())
                        
                        fig = px.bar(x=model_names, y=inference_times, 
                                   title='Inference Time Comparison',
                                   labels={'x': 'Model', 'y': 'Inference Time (s)'})
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # FPS comparison
                        fps_values = [metrics['fps'] for metrics in benchmark_results['benchmark_results'].values()]
                        
                        fig = px.bar(x=model_names, y=fps_values,
                                   title='FPS Comparison',
                                   labels={'x': 'Model', 'y': 'FPS'})
                        st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.error(f"❌ Benchmark failed: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


def model_management_page():
    """Model management and configuration"""
    
    st.header("🛠️ Model Management")
    
    # Get current model info
    try:
        response = requests.get(f"{API_BASE_URL}/models/info", timeout=5)
        if response.status_code == 200:
            models_info = response.json()
            
            st.subheader("📋 Current Configuration")
            
            # Display ensemble method
            st.write(f"**Ensemble Method**: {models_info.get('ensemble_method', 'N/A')}")
            
            # Display model weights
            st.subheader("⚖️ Model Weights")
            
            current_weights = models_info.get('weights', {})
            
            # Weight adjustment
            new_weights = {}
            
            for model_name, current_weight in current_weights.items():
                new_weights[model_name] = st.slider(
                    f"{model_name} Weight",
                    0.0, 2.0, current_weight, 0.1,
                    key=f"weight_{model_name}"
                )
            
            # Update weights
            if st.button("🔄 Update Weights"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/models/ensemble/update_weights",
                        json=new_weights,
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Weights updated successfully!")
                        
                        # Display updated weights
                        st.write("**Updated Weights:**")
                        for model_name, weight in result['updated_weights'].items():
                            st.write(f"- {model_name}: {weight:.3f}")
                    else:
                        st.error(f"❌ Weight update failed: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
            
            # Model information
            st.subheader("🔍 Model Information")
            
            for model_name, model_info in models_info.get('models', {}).items():
                with st.expander(f"{model_name} Details"):
                    st.write(f"**Enabled**: {model_info.get('enabled', False)}")
                    st.write(f"**Architecture**: {model_info.get('architecture', 'N/A')}")
                    st.write(f"**Weight**: {model_info.get('weight', 'N/A')}")
        
        else:
            st.error("❌ Cannot fetch model information")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def training_dashboard_page():
    """Training dashboard and monitoring"""
    
    st.header("📈 Training Dashboard")
    
    # Placeholder for training metrics
    st.info("🚧 Training dashboard is under development")
    
    # Mock training data for demonstration
    st.subheader("📊 Training Metrics")
    
    # Generate mock data
    epochs = list(range(1, 51))
    train_loss = [0.8 - 0.01 * i + 0.1 * np.sin(i * 0.2) for i in epochs]
    val_loss = [0.9 - 0.008 * i + 0.15 * np.sin(i * 0.3) for i in epochs]
    
    # Create training curves
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=epochs, y=train_loss,
        mode='lines',
        name='Training Loss',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=epochs, y=val_loss,
        mode='lines',
        name='Validation Loss',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Training Progress',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Domain adaptation metrics
    st.subheader("🔄 Domain Adaptation Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Source Accuracy", "92.3%", "↑ 2.1%")
        st.metric("Target Accuracy", "87.8%", "↑ 5.2%")
    
    with col2:
        st.metric("Domain Accuracy", "78.9%", "↑ 3.4%")
        st.metric("Adaptation Gap", "4.5%", "↓ 1.8%")


def visualize_detections(image, predictions):
    """Visualize detection results on image"""
    
    result_image = image.copy()
    
    boxes = predictions.get('boxes', [])
    scores = predictions.get('scores', [])
    classes = predictions.get('classes', [])
    
    # Define colors for different classes
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Red
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
    ]
    
    for i, (box, score, class_id) in enumerate(zip(boxes, scores, classes)):
        if score > 0.3:  # Confidence threshold
            x1, y1, x2, y2 = [int(coord) for coord in box]
            color = colors[int(class_id) % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Class {int(class_id)}: {score:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_image


if __name__ == "__main__":
    main() 