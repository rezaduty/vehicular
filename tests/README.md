# Tests & Demos

## Core Tests
- `test_project.py` - Comprehensive project functionality test
- `test_configuration_functionality.py` - Configuration system tests
- `test_parameter_functionality.py` - Parameter control tests
- `test_real_functionality.py` - Real model functionality tests

## Specific Feature Tests
- `test_detection_modes.py` - Detection modes and patch processing
- `test_real_domain_adaptation.py` - Domain adaptation training
- `test_streamlit_config_integration.py` - Web interface integration

## Demo Scripts
- `demo.py` - Basic API demonstration
- `simple_demo.py` - Simple detection demo
- `demo_upload_test.py` - Image upload testing
- `generate_test_images.py` - Test image generation

## Running Tests

```bash
# Run main project test
python tests/test_project.py

# Test configuration functionality
python tests/test_configuration_functionality.py

# Test all detection modes
python tests/test_detection_modes.py

# Test Streamlit integration
python tests/test_streamlit_config_integration.py
```

All tests assume the API server is running on `http://localhost:8000`. 