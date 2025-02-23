# tests/test_api.py
import pytest
import io
from PIL import Image
import numpy as np
from app import create_app
from unittest.mock import patch, MagicMock
import os
import tensorflow as tf

@pytest.fixture
def app():
    """Create and configure app for testing."""
    app = create_app()
    app.config['TESTING'] = True
    app.config['MODEL_PATH'] = 'models/facenet_real_fake_classifier_final.keras'
    return app

@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()

def create_test_image():
    """Create a test image for API testing."""
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
            'confidence': 0.2,  # Less than 0.5 means fake
            'processing_time': 0.1,
            'debug_info': {
                'input_shape': [1, 160, 160, 3],
                'raw_prediction': 0.2
            }
        }
        yield mock

@pytest.fixture
def mock_r2_storage():
    """Mock R2 storage for testing."""
    with patch('app.utils.r2_storage.R2ModelStorage') as mock:
        instance = mock.return_value
        instance.download_model.return_value = 'models/facenet_real_fake_classifier_final.keras'
        yield mock

def test_deepfake_detection_endpoint(client, mock_model, mock_r2_storage):
    """Test the deepfake detection endpoint."""
    test_image = create_test_image()
    
    response = client.post(
        '/api/v1/detect/',
        data={'file': (test_image, 'test.png')},
        content_type='multipart/form-data'
    )
    
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

def test_health_check(client, mock_r2_storage):
    """Test the health check endpoint."""
    response = client.get('/api/v1/detect/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_augmented_detection(client, mock_model, mock_r2_storage):
    """Test the augmented detection endpoint."""
    test_image = create_test_image()
    
    response = client.post(
        '/api/v1/detect/augmented',
        data={'file': (test_image, 'test.png')},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    
    assert 'original_prediction' in data
    assert 'augmented_predictions' in data
    assert 'consensus_prediction' in data
    assert isinstance(data['augmented_predictions'], list)
    assert len(data['augmented_predictions']) > 0

def test_model_download():
    """Test R2 model download functionality."""
    from app.utils.r2_storage import R2ModelStorage
    
    # Create instance with test credentials
    r2_storage = R2ModelStorage()
    
    # Test if model exists or can be downloaded
    model_path = r2_storage.local_model_path
    assert os.path.exists(model_path), "Model file should exist after download"