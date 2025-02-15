# app/utils/image_processing.py
def validate_image(file):
    """Validate uploaded image file"""
    # Check if file exists
    if not file:
        raise ValueError("No file provided")
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        raise ValueError("Invalid file type. Allowed types: PNG, JPG, JPEG")
    
    # You might want to add more validations (file size, image dimensions, etc.)
    return True