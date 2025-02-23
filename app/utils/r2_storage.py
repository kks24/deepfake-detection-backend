# app/utils/r2_storage.py
import boto3
import os
import logging
from botocore.config import Config

class R2ModelStorage:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('R2_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY'),
            config=Config(
                retries={'max_attempts': 3},
                connect_timeout=5,
                read_timeout=300
            )
        )
        self.bucket_name = os.getenv('R2_BUCKET_NAME')
        self.model_key = os.getenv('R2_MODEL_KEY')
        self.local_model_path = os.path.join('models', 'facenet_real_fake_classifier_final.keras')
        
    def verify_r2_model(self):
        """Verify model exists in R2 and get its size"""
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=self.model_key
            )
            expected_size = response['ContentLength']
            logging.info(f"R2 model size: {expected_size / (1024*1024):.2f} MB")
            return expected_size
        except Exception as e:
            logging.error(f"Error verifying R2 model: {str(e)}")
            raise

    def verify_local_model(self, expected_size=None):
        """Verify local model file size matches R2"""
        if not os.path.exists(self.local_model_path):
            return False
            
        actual_size = os.path.getsize(self.local_model_path)
        if expected_size and actual_size != expected_size:
            logging.warning(f"Local model size ({actual_size} bytes) does not match R2 ({expected_size} bytes)")
            return False
            
        return True

    def download_model(self):
        """Downloads model from R2 with verification"""
        try:
            # First verify R2 model
            expected_size = self.verify_r2_model()
            
            # Check if we already have a valid local copy
            if self.verify_local_model(expected_size):
                logging.info("Using existing local model file")
                return self.local_model_path
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.local_model_path), exist_ok=True)
            
            logging.info(f"Downloading model from R2: {self.model_key}")
            
            # Download with progress tracking
            def progress_callback(bytes_transferred):
                mb_transferred = bytes_transferred / (1024*1024)
                logging.info(f"Downloaded: {mb_transferred:.2f} MB")
            
            # Download file
            self.s3_client.download_file(
                self.bucket_name,
                self.model_key,
                self.local_model_path,
                Callback=progress_callback
            )
            
            # Verify downloaded file
            if not self.verify_local_model(expected_size):
                raise Exception("Downloaded file size does not match R2")
                
            logging.info(f"Model downloaded successfully to {self.local_model_path}")
            return self.local_model_path

        except Exception as e:
            logging.error(f"Error downloading model from R2: {str(e)}")
            # Clean up partial download if it exists
            if os.path.exists(self.local_model_path):
                os.remove(self.local_model_path)
            raise