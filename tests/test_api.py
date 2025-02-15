# tests/test_api.py
import pytest
from app import create_app
import io
from PIL import Image
import numpy as np

@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def create_test_image():
    """Create a test image for API testing"""
    file = io.BytesIO()
    image = Image.new('RGB', (224, 224), color='red')
    image.save(file, 'PNG')
    file.seek(0)
    return file

def test_deepfake_detection_endpoint(client):
    """Test the deepfake detection endpoint"""
    test_image = create_test_image()
    response = client.post(
        '/api/v1/detect/',
        data={'file': (test_image, 'test.png')},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 200
    assert 'is_fake' in response.json
    assert 'confidence' in response.json
    assert 'processing_time' in response.json