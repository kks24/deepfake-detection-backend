# test_deepfake.py
import tensorflow as tf
import logging
from app.models.deepfake_model import DeepfakeDetector
from PIL import Image
import sys

logging.basicConfig(level=logging.INFO)

def test_model():
    print(f"TensorFlow version: {tf.__version__}")
    print("Initializing detector...")
    
    try:
        detector = DeepfakeDetector()
        print("Detector initialized successfully!")
        return detector
    except Exception as e:
        print(f"Error initializing detector: {e}")
        return None

def test_prediction(detector, image_path):
    try:
        # Load and process image
        with Image.open(image_path) as img:
            result = detector.predict(img)
            
        print("\nPrediction results:")
        for key, value in result.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    detector = test_model()
    if detector and len(sys.argv) > 1:
        test_prediction(detector, sys.argv[1])
    else:
        print("Usage: python test_deepfake.py path/to/image.jpg")
