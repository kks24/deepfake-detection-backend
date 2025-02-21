# app/models/deepfake_model.py
import tensorflow as tf
import numpy as np
import os
import logging
from PIL import Image
import cv2
import time
import gc
from tensorflow.keras.models import load_model 
from tensorflow.keras.preprocessing import image
import keras

class DeepfakeDetector:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeepfakeDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.input_shape = (150, 150)  # Base shape for internal use
        self.image_size = (160, 160)   # Size for preprocessing - matching training
        self._initialized = True
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.configure_tensorflow()
    
    def configure_tensorflow(self):
        try:
            tf.config.set_visible_devices([], 'GPU')
            tf.config.run_functions_eagerly(True)
        except Exception as e:
            self.logger.warning(f"Error configuring TensorFlow: {str(e)}")
    
    def load_model(self):
        if self._model is None:
            try:
                # Clear any existing sessions
                tf.keras.backend.clear_session()
                gc.collect()

                # Register custom functions
                @keras.saving.register_keras_serializable()
                def scaling(x, scale=255.0, **kwargs):
                    return x * scale

                @keras.saving.register_keras_serializable()
                def l2_normalize(x, axis=-1, epsilon=1e-10):
                    return tf.nn.l2_normalize(x, axis=axis, epsilon=epsilon)

                # Define custom objects with the registered functions
                custom_objects = {
                    "scaling": scaling, 
                    "l2_normalize": l2_normalize
                }
                
                # Load model with custom objects
                model_path = os.getenv('MODEL_PATH', 'models/facenet_real_fake_classifier_final.keras')
                self._model = load_model(
                    model_path,
                    custom_objects=custom_objects
                )
                
                self.logger.info(f"Model loaded successfully. Input shape: {self._model.input_shape}")
                
            except Exception as e:
                self.logger.error(f"Error loading model: {str(e)}")
                raise
    
    def preprocess_image(self, image_file):
        """
        Preprocess image following the exact training pipeline
        """
        try:
            # Open and resize image
            img = Image.open(image_file)
            img = img.resize(self.image_size)  # Using 160x160 as in training
            img = img.convert("RGB")  # Ensure RGB format
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            self.logger.info(f"Preprocessed image shape: {img_array.shape}")
            return img_array
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def predict(self, image):
        """
        Make prediction on an image using the loaded model
        """
        try:
            start_time = time.time()
            
            # Ensure model is loaded
            if self._model is None:
                self.load_model()

            # Preprocess image
            img_array = self.preprocess_image(image)

            # Define label mapping
            label_map = {0: "fake", 1: "real"}

            # Make prediction
            with tf.device('/CPU:0'):
                pred_prob = self._model.predict(img_array)[0][0]

            # Convert prediction to label
            pred_label = int(pred_prob > 0.5)
            confidence = float(pred_prob)

            # Return formatted result
            result = {
                "prediction": label_map[pred_label],
                "confidence": confidence,
                "processing_time": time.time() - start_time,
                "debug_info": {
                    "input_shape": list(img_array.shape),
                    "raw_prediction": float(pred_prob)
                }
            }
            
            self.logger.info(f"Prediction result: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self._model is None:
            self.load_model()
        return {
            'input_shape': list(self.input_shape),
            'image_size': list(self.image_size),
            'model_path': './models/facenet_real_fake_classifier_final.keras',
            'layers': [layer.name for layer in self._model.layers]
        }
