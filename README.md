# Autonomous Driving Perception System

Real-time object detection system with parallel patch processing, domain adaptation, and configuration management for autonomous vehicles.

## Quick Start

### Prerequisites
```bash
python3.8+
pip
```

### Installation
```bash
# Clone and navigate
git clone <repository-url>
cd vehi

# Install dependencies
pip install -r requirements.txt

# Download YOLOv8 model (if not present)
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### Run Application

1. **Start API Server**
   ```bash
   uvicorn src.api.real_working_api:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Start Web Interface** (new terminal)
   ```bash
   streamlit run src/streamlit_app.py --server.port 8501
   ```

3. **Access Application**
   - Web UI: http://localhost:8501
   - API: http://localhost:8000

## Features

- **Object Detection**: Real YOLOv8-based detection
- **Patch Detection**: Enhanced small object detection  
- **Domain Adaptation**: Simulation to real-world transfer
- **Configuration**: Real-time parameter updates
- **Unsupervised Detection**: LOST algorithm implementation

## Project Structure

```
├── src/                # Source code
│   ├── api/           # FastAPI backend
│   ├── models/        # ML models
│   ├── data/          # Data processing
│   └── streamlit_app.py
├── tests/             # All tests and demos
├── docs/              # Documentation
├── config/            # Configuration files
└── requirements.txt   # Dependencies
```

## Usage

1. Open web interface at http://localhost:8501
2. Upload images for object detection
3. Try different detection modes (standard/patch)
4. Configure system parameters in Settings
5. Test domain adaptation training

## Testing

```bash
# Run comprehensive tests
python tests/test_project.py

# Test configuration functionality
python tests/test_configuration_functionality.py

# Test detection modes
python tests/test_detection_modes.py
```

## API Endpoints

- `GET /health` - System health
- `POST /detect` - Object detection
- `POST /domain_adapt` - Start domain adaptation
- `GET/POST /config` - Configuration management

## Documentation

See `docs/` folder for detailed documentation:
- Configuration management
- Domain adaptation explained
- Real functionality verification
- Testing guides

---

**Developed for Vehicular Technology Project**  
*Advanced perception system for autonomous driving applications* 