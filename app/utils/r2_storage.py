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
                read_timeout=300  # Longer timeout for large files
            )
        )
        self.bucket_name = os.getenv('R2_BUCKET_NAME')
        self.model_key = os.getenv('R2_MODEL_KEY')
        self.local_model_path = os.path.join('models', 'facenet_real_fake_classifier_final.keras')

    def download_model(self):
        """Downloads the model file from R2 to local storage"""
        try:
            os.makedirs('models', exist_ok=True)
            
            logging.info(f"Downloading model from R2: {self.model_key}")
            
            self.s3_client.download_file(
                self.bucket_name,
                self.model_key,
                self.local_model_path
            )
            
            logging.info(f"Model downloaded successfully to {self.local_model_path}")
            return self.local_model_path

        except Exception as e:
            logging.error(f"Error downloading model from R2: {str(e)}")
            raise