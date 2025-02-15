# app/api/v1/routes.py
from flask import Blueprint, request, current_app, jsonify
from flask_restx import Api, Resource, fields
from werkzeug.datastructures import FileStorage
from datetime import datetime
import logging
from PIL import Image, ImageOps
from app.models.deepfake_model import DeepfakeDetector
import io
import base64, time
import numpy as np

# Initialize Blueprint and API
api_v1 = Blueprint('api_v1', __name__)
api = Api(api_v1, 
          version='1.0', 
          title='Deepfake Detection API',
          description='API for detecting fake faces using deep learning',
          doc='/docs')

# Define namespaces
detect_ns = api.namespace('detect', description='Deepfake detection operations')
model_ns = api.namespace('model', description='Model information and operations')

# Define response models
prediction_response = api.model('PredictionResponse', {
    'prediction': fields.String(description='Prediction result (REAL/FAKE)',
                              example='FAKE'),
    'confidence': fields.Float(description='Confidence score',
                             example=0.95),
    'processing_time': fields.Float(description='Time taken to process in seconds',
                                  example=0.534),
    'debug_info': fields.Raw(description='Additional debug information',
                           example={
                               'input_shape': [1, 160, 160, 3],
                               'raw_prediction': 0.95
                           })
})

augmented_prediction = api.model('AugmentedPrediction', {
    'augmentation_type': fields.String(description='Type of augmentation applied'),
    'prediction': fields.String(description='Prediction (REAL/FAKE)'),
    'confidence': fields.Float(description='Confidence score')
})

augmented_response = api.model('AugmentedResponse', {
    'original_prediction': fields.Raw(description='Original image prediction'),
    'augmented_predictions': fields.List(fields.Nested(augmented_prediction)),
    'consensus_prediction': fields.String(description='Final prediction based on all augmentations'),
    'average_confidence': fields.Float(description='Average confidence across all predictions'),
    'processing_time': fields.Float(description='Total processing time in seconds'),
    'debug_info': fields.Raw(description='Additional debug information')
})

error_response = api.model('ErrorResponse', {
    'error': fields.String(description='Error message'),
    'error_code': fields.String(description='Error code for reference'),
    'timestamp': fields.DateTime(description='Time of error')
})

model_info_response = api.model('ModelInfo', {
    'input_shape': fields.List(fields.Integer, description='Model input shape'),
    'image_size': fields.List(fields.Integer, description='Expected image size'),
    'model_path': fields.String(description='Path to the model file'),
    'layers': fields.List(fields.String, description='Model layer names')
})

# File upload parser
upload_parser = api.parser()
upload_parser.add_argument('file',
                        location='files',
                        type=FileStorage,
                        required=True,
                        help='Image file to analyze (JPEG/PNG)')

@detect_ns.route('/')
class DeepfakeDetection(Resource):
    @api.expect(upload_parser)
    @api.response(200, 'Success', prediction_response)
    @api.response(400, 'Invalid input', error_response)
    @api.response(500, 'Server error', error_response)
    def post(self):
        """
        Submit an image for deepfake detection analysis.
        The API accepts an image file and returns prediction results including:
        - Prediction (REAL/FAKE)
        - Confidence score
        - Processing time
        - Debug information
        """
        try:
            # Validate request
            if 'file' not in request.files:
                return {
                    'error': 'No file provided',
                    'error_code': 'MISSING_FILE',
                    'timestamp': datetime.utcnow().isoformat()
                }, 400
            
            file = request.files['file']
            if not file.filename:
                return {
                    'error': 'No selected file',
                    'error_code': 'EMPTY_FILE',
                    'timestamp': datetime.utcnow().isoformat()
                }, 400
            
            # Process image and get prediction
            try:
                detector = DeepfakeDetector()
                result = detector.predict(file)
                return result, 200
                
            except Exception as e:
                current_app.logger.error(f"Error processing image: {str(e)}")
                return {
                    'error': 'Error processing image',
                    'error_code': 'PROCESSING_ERROR',
                    'details': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }, 500
                
        except Exception as e:
            current_app.logger.error(f"Server error: {str(e)}")
            return {
                'error': 'Internal server error',
                'error_code': 'SERVER_ERROR',
                'timestamp': datetime.utcnow().isoformat()
            }, 500

@model_ns.route('/info')
class ModelInfo(Resource):
    @api.response(200, 'Success', model_info_response)
    @api.response(500, 'Server error', error_response)
    def get(self):
        """
        Get information about the currently loaded model.
        Returns details about the model architecture, input requirements,
        and configuration.
        """
        try:
            detector = DeepfakeDetector()
            info = detector.get_model_summary()
            return info, 200
        except Exception as e:
            return {
                'error': 'Error retrieving model information',
                'error_code': 'MODEL_INFO_ERROR',
                'details': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, 500

@detect_ns.route('/health')
class HealthCheck(Resource):
    @api.response(200, 'Success')
    @api.response(503, 'Service unhealthy')
    def get(self):
        """
        Check if the API and model are operational.
        Verifies that the model can be loaded and the service is running properly.
        """
        try:
            detector = DeepfakeDetector()
            detector.load_model()  # Verify model can be loaded
            
            return {
                'status': 'healthy',
                'model_loaded': True,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }, 503
            
class ImageAugmenter:
    """Handles various image augmentation techniques"""
    
    def __init__(self):
        self.augmentations = {
            'original': lambda img: img,
            'grayscale': self.to_grayscale,
            'rotate_right': lambda img: img.rotate(-20),
            'rotate_left': lambda img: img.rotate(20),
            'brightness_up': lambda img: ImageOps.autocontrast(img, cutoff=10),
            'brightness_down': lambda img: img.point(lambda p: p * 0.8),
            'flip_horizontal': lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)
        }
    
    def to_grayscale(self, img):
        """Convert image to grayscale while maintaining RGB format"""
        gray = img.convert('L')
        return Image.merge('RGB', (gray, gray, gray))
    
    def apply_augmentations(self, image):
        """Apply all defined augmentations to an image"""
        results = {}
        for name, aug_func in self.augmentations.items():
            try:
                augmented = aug_func(image)
                # Ensure image is in correct size
                if augmented.size != (160, 160):
                    augmented = augmented.resize((160, 160))
                results[name] = augmented
            except Exception as e:
                current_app.logger.error(f"Error applying {name} augmentation: {str(e)}")
        return results

@detect_ns.route('/augmented')
class AugmentedDetection(Resource):
    @api.expect(upload_parser)
    @api.response(200, 'Success', augmented_response)
    @api.response(400, 'Invalid input', error_response)
    @api.response(500, 'Server error', error_response)
    def post(self):
        """
        Submit an image for augmented deepfake detection analysis.
        Applies various data augmentation techniques and aggregates predictions.
        """
        try:
            start_time = time.time()
            
            # Validate request
            if 'file' not in request.files:
                return {
                    'error': 'No file provided',
                    'error_code': 'MISSING_FILE',
                    'timestamp': datetime.utcnow().isoformat()
                }, 400
            
            file = request.files['file']
            if not file.filename:
                return {
                    'error': 'No selected file',
                    'error_code': 'EMPTY_FILE',
                    'timestamp': datetime.utcnow().isoformat()
                }, 400
            
            try:
                image = Image.open(file).convert('RGB')
                image = image.resize((160, 160))
                
                augmenter = ImageAugmenter()
                detector = DeepfakeDetector()
                
                augmented_images = augmenter.apply_augmentations(image)
                
                predictions = []
                all_confidences = []
                low_confidence_count = 0  # Count of predictions with confidence < 0.5
                
                # Process each augmented version
                for aug_type, aug_image in augmented_images.items():
                    img_byte_arr = io.BytesIO()
                    aug_image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    
                    result = detector.predict(img_byte_arr)
                    confidence = result['confidence']
                    
                    # Store prediction
                    pred_entry = {
                        'augmentation_type': aug_type,
                        'prediction': result['prediction'],
                        'confidence': confidence
                    }
                    
                    # Track statistics
                    if aug_type == 'original':
                        original_result = result
                    else:
                        predictions.append(pred_entry)
                    
                    all_confidences.append(confidence)
                    if confidence < 0.5:  # Using 0.5 as threshold
                        low_confidence_count += 1
                
                # Calculate consensus based on number of low confidence predictions
                total_predictions = len(predictions) + 1  # +1 for original
                consensus = 'FAKE' if low_confidence_count >= (total_predictions / 2) else 'REAL'
                
                # Calculate overall average confidence
                avg_confidence = float(np.mean(all_confidences))
                
                response = {
                    'original_prediction': original_result,
                    'augmented_predictions': predictions,
                    'consensus_prediction': consensus,
                    'average_confidence': avg_confidence,
                    'processing_time': float(time.time() - start_time),
                    'debug_info': {
                        'predictions_below_threshold': low_confidence_count,
                        'total_predictions': total_predictions,
                        'threshold': 0.5,
                        'input_size': image.size
                    }
                }
                
                return response, 200
                
            except Exception as e:
                current_app.logger.error(f"Error processing augmented detection: {str(e)}")
                return {
                    'error': 'Error processing image',
                    'error_code': 'PROCESSING_ERROR',
                    'details': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }, 500
                
        except Exception as e:
            current_app.logger.error(f"Server error in augmented detection: {str(e)}")
            return {
                'error': 'Internal server error',
                'error_code': 'SERVER_ERROR',
                'timestamp': datetime.utcnow().isoformat()
            }, 500
