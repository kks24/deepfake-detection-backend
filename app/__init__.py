# app/__init__.py
from flask import Flask
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
from app.core.logging import setup_logging
from app.models.deepfake_model import DeepfakeDetector

def create_app():
    app = Flask(__name__)
    CORS(app)
    
    # Initialize rate limiter
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"]
    )
    
    # Setup logging
    setup_logging(app)
    
    # Initialize model at startup
    try:
        DeepfakeDetector()
        app.logger.info("Model initialized successfully")
    except Exception as e:
        app.logger.error(f"Error initializing model: {str(e)}")
    
    # Register blueprints
    from app.api.v1.routes import api_v1
    app.register_blueprint(api_v1, url_prefix='/api/v1')
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)