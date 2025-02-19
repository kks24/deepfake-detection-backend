# tests/test_api.py
import pytest
import io
from PIL import Image
import numpy as np
from app import create_app
from unittest.mock import patch, MagicMock

@pytest.fixture
def app():
    """Create and configure app for testing."""
    app = create_app()
    app.config['TESTING'] = True
    app.config['MODEL_PATH'] = 'test_model.h5'  # Use a test model path
    return app

@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()

def create_test_image():
    """Create a test image for API testing."""
    # Create a 160x160 RGB test image
    image = Image.new('RGB', (160, 160), color='red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def mock_model():
    """Create a mock model that returns predictable results."""
    with patch('app.models.deepfake_model.DeepfakeDetector') as mock:
        instance = mock.return_value
        instance.predict.return_value = {
            'prediction': 'fake',
            'confidence': 0.8,
            'processing_time': 0.1,
            'debug_info': {
                'input_shape': [1, 160, 160, 3],
                'raw_prediction': 0.8
            }
        }
        yield mock

def test_deepfake_detection_endpoint(client, mock_model):
    """Test the deepfake detection endpoint with a mock model."""
    # Create test image
    test_image = create_test_image()
    
    # Make request to the API
    response = client.post(
        '/api/v1/detect/',
        data={'file': (test_image, 'test.png')},
        content_type='multipart/form-data'
    )
    
    # Check response
    assert response.status_code == 200
    data = response.get_json()
    
    # Verify response structure
    assert 'prediction' in data
    assert 'confidence' in data
    assert 'processing_time' in data
    assert 'debug_info' in data
    
    # Verify expected values
    assert data['prediction'] == 'fake'
    assert isinstance(data['confidence'], float)
    assert isinstance(data['processing_time'], float)

def test_missing_file(client):
    """Test the API's response when no file is provided."""
    response = client.post(
        '/api/v1/detect/',
        data={},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == 'No file provided'

def test_empty_file(client):
    """Test the API's response when an empty file is provided."""
    response = client.post(
        '/api/v1/detect/',
        data={'file': (io.BytesIO(), '')},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == 'No selected file'

def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get('/api/v1/detect/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_augmented_detection_endpoint(client, mock_model):
    """Test the augmented detection endpoint with a mock model."""
    test_image = create_test_image()
    
    response = client.post(
        '/api/v1/detect/augmented',
        data={'file': (test_image, 'test.png')},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    
    # Verify augmented response structure
    assert 'original_prediction' in data
    assert 'augmented_predictions' in data
    assert 'consensus_prediction' in data
    assert 'average_confidence' in data
    assert isinstance(data['augmented_predictions'], list)
    assert len(data['augmented_predictions']) > 0