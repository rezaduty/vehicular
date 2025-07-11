#!/bin/bash

echo "🚀 Starting Autonomous Driving Perception System..."
echo "======================================================"

# Kill existing processes
echo "🔄 Stopping existing services..."
pkill -f "uvicorn.*real_working_api" 2>/dev/null
pkill -f "streamlit.*streamlit_app" 2>/dev/null

# Wait a moment for processes to stop
sleep 2

# Start API server in background
echo "🔧 Starting API server..."
uvicorn src.api.real_working_api:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# Wait for API to start
sleep 5

# Check if API is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API server started successfully on http://localhost:8000"
else
    echo "❌ API server failed to start"
    exit 1
fi

# Start Streamlit app
echo "🎨 Starting Streamlit interface..."
streamlit run src/streamlit_app.py --server.port 8501 &
STREAMLIT_PID=$!

# Wait for Streamlit to start
sleep 3

echo "======================================================"
echo "🎉 Services started successfully!"
echo "📊 API Documentation: http://localhost:8000/docs"
echo "🎨 Streamlit Interface: http://localhost:8501"
echo "======================================================"
echo "Available Models:"
echo "  🤖 YOLOv8 - Fast object detection"
echo "  🔄 Domain Adaptation - DANN method"
echo "  🔍 Patch Detection - Enhanced small objects"
echo "  🎯 Unsupervised LOST - Self-supervised learning"
echo "======================================================"
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping services..."
    kill $API_PID $STREAMLIT_PID 2>/dev/null
    echo "✅ Services stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Wait for processes
wait 