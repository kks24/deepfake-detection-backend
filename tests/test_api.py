# tests/test_api.py
import pytest
import io
from PIL import Image
import numpy as np
from app import create_app
from unittest.mock import patch, MagicMock
import boto3
import os

@pytest.fixture(scope='session', autouse=True)
def download_model():
    """Download model from R2 before running tests"""
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('R2_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY')
        )
        
        os.makedirs('models', exist_ok=True)
        model_path = 'models/facenet_real_fake_classifier_final.keras'
        
        if not os.path.exists(model_path):
            s3_client.download_file(
                os.getenv('R2_BUCKET_NAME'),
                os.getenv('R2_MODEL_KEY'),
                model_path
            )
    except Exception as e:
        pytest.skip(f"Failed to download model: {str(e)}")

@pytest.fixture
def app():
    """Create and configure app for testing"""
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    """Create a test client"""
    return app.test_client()

def create_test_image():
    """Create a test image for API testing"""
    # Create a 160x160 RGB test image
    image = Image.new('RGB', (160, 160), color='red')
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get('/api/v1/detect/health')
    assert response.status_code == 200
    data = response.get_json()
    assert data['status'] == 'healthy'

def test_detect_endpoint_with_valid_image(client):
    """Test the detection endpoint with a valid image"""
    # Create and send test image
    test_image = create_test_image()
    response = client.post(
        '/api/v1/detect/',
        data={'file': (test_image, 'test.png')},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    
    # Check response structure
    assert 'prediction' in data
    assert 'confidence' in data
    assert isinstance(data['confidence'], float)
    assert data['prediction'] in ['REAL', 'FAKE']

def test_detect_endpoint_without_file(client):
    """Test the detection endpoint without providing a file"""
    response = client.post(
        '/api/v1/detect/',
        data={},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == 'No file provided'

def test_detect_endpoint_with_empty_file(client):
    """Test the detection endpoint with an empty file"""
    response = client.post(
        '/api/v1/detect/',
        data={'file': (io.BytesIO(), '')},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data
    assert data['error'] == 'No selected file'

def test_augmented_detect_endpoint(client):
    """Test the augmented detection endpoint"""
    test_image = create_test_image()
    response = client.post(
        '/api/v1/detect/augmented',
        data={'file': (test_image, 'test.png')},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 200
    data = response.get_json()
    
    # Check response structure
    assert 'original_prediction' in data
    assert 'augmented_predictions' in data
    assert 'consensus_prediction' in data
    assert 'average_confidence' in data
    assert isinstance(data['augmented_predictions'], list)
    assert len(data['augmented_predictions']) > 0

def test_model_info_endpoint(client):
    """Test the model info endpoint"""
    response = client.get('/api/v1/model/info')
    
    assert response.status_code == 200
    data = response.get_json()
    
    # Check response structure
    assert 'input_shape' in data
    assert 'image_size' in data
    assert 'model_path' in data