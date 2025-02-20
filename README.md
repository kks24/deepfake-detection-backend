# Deepfake Detection Backend

This is the backend service for the Deepfake Detection system, built using Flask and TensorFlow. The service provides API endpoints for detecting fake faces in images using deep learning.

## Features

- Image-based deepfake detection
- Multiple detection methods including standard and augmented detection
- RESTful API with Swagger documentation
- Docker containerization
- Health monitoring endpoints
- Automated CI/CD pipeline

## Tech Stack

- Python 3.11
- Flask & Flask-RESTX
- TensorFlow
- Docker
- GitHub Actions for CI/CD
- Gunicorn for WSGI server

## API Endpoints

### Main Endpoints

- `POST /api/v1/detect/` - Main detection endpoint
- `POST /api/v1/detect/augmented` - Augmented detection with multiple image transformations
- `GET /api/v1/detect/health` - Health check endpoint
- `GET /api/v1/model/info` - Model information endpoint

### API Documentation

API documentation is available via Swagger UI at `/api/v1/docs` when the service is running.

## Getting Started

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Model file (facenet_real_fake_classifier_2Stage_2.h5)

### Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/deepfake-detection-backend.git
cd deepfake-detection-backend
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export FLASK_APP=app/__init__.py
export FLASK_ENV=development
export MODEL_PATH=models/facenet_real_fake_classifier_final.keras
```

5. Run the development server:
```bash
flask run
```

### Docker Setup

1. Build the Docker image:
```bash
docker build -t deepfake-detection-backend .
```

2. Run with Docker Compose:
```bash
docker-compose up
```

## Testing

Run the test suite:
```bash
python -m pytest
```

## API Usage Examples

### Detection Endpoint

```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:5000/api/v1/detect/
```

Example Response:
```json
{
    "prediction": "fake",
    "confidence": 0.95,
    "processing_time": 0.534,
    "debug_info": {
        "input_shape": [1, 160, 160, 3],
        "raw_prediction": 0.95
    }
}
```

### Augmented Detection

```bash
curl -X POST -F "file=@/path/to/image.jpg" http://localhost:5000/api/v1/detect/augmented
```

### Health Check

```bash
curl http://localhost:5000/api/v1/detect/health
```

## Deployment

The service is set up for automatic deployment using GitHub Actions. The pipeline:
1. Runs tests
2. Builds Docker image
3. Pushes to Docker Hub
4. Deploys to production server

### Environment Variables

Required environment variables for deployment:
- `FLASK_APP`: Application entry point
- `FLASK_ENV`: Environment (production/development)
- `MODEL_PATH`: Path to the model file
- `DOCKER_HUB_USERNAME`: Docker Hub username
- `DOCKER_HUB_TOKEN`: Docker Hub access token
- `SERVER_HOST`: Deployment server host
- `SERVER_USERNAME`: Server SSH username
- `SERVER_SSH_KEY`: Server SSH private key
