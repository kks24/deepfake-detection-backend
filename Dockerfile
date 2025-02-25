# Use a multi-arch base image
FROM --platform=$TARGETPLATFORM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for multiple architectures
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p logs

# Set environment variables
ENV FLASK_APP=app/__init__.py
ENV FLASK_ENV=production
ENV MODEL_PATH=models/facenet_real_fake_classifier_final.keras

# Expose the port
EXPOSE 5000

# Command to run the application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]

