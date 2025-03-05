import os
import sys
import torch
import timm
import logging
import traceback
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.StreamHandler(sys.stderr)
                    ])
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Model Configuration
CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
MODEL_PATH = 'student_model.pth'
IMAGE_SIZE = 224

class BloodCellClassifier:
    def __init__(self, model_path, classes):
        try:
            logger.info(f"Initializing model from {model_path}")
            
            # Verify model file exists
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load the Vision Transformer model
            self.model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=len(classes))
            
            # Load pre-trained weights
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            
            # Log model state dict keys for debugging
            logger.info(f"Loaded state dict keys: {list(state_dict.keys())}")
            
            self.model.load_state_dict(state_dict)
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.classes = classes
            
            # Image transformation pipeline
            self.transform = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Model initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict_class(self, image):
        """
        Predict the class of the input image
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            tuple: Predicted class name and confidence score
        """
        try:
            # Preprocess the image
            input_tensor = self.transform(image).unsqueeze(0)
            
            # Disable gradient computation for inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                class_name = self.classes[predicted_class.item()]
                confidence_score = confidence.item() * 100
                
                logger.info(f"Prediction: {class_name} with {confidence_score:.2f}% confidence")
                
                return class_name, confidence_score
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            raise

# Initialize the classifier
try:
    classifier = BloodCellClassifier(MODEL_PATH, CLASSES)
except Exception as e:
    logger.critical(f"Failed to initialize classifier: {str(e)}")
    classifier = None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if classifier is None:
        return jsonify({'error': 'Classifier not initialized'}), 500
    
    if 'file' not in request.files:
        logger.warning("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Open the image
        image = Image.open(file.stream).convert('RGB')
        logger.info(f"Received image: {file.filename}")
        
        # Predict the class
        predicted_class, confidence = classifier.predict_class(image)
        
        return jsonify({
            'class': predicted_class,
            'confidence': f'{confidence:.2f}%'
        })
    
    except Exception as e:
        logger.error(f"Prediction route error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Error processing image',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
