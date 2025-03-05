import os
import torch
import timm
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Model Configuration
CLASSES = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']
MODEL_PATH = 'student_model.pth'
IMAGE_SIZE = 224

class BloodCellClassifier:
    def __init__(self, model_path, classes):
        # Load the Vision Transformer model
        self.model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=len(classes))
        
        # Load pre-trained weights
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
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
    
    def predict_class(self, image):
        """
        Predict the class of the input image
        
        Args:
            image (PIL.Image): Input image
        
        Returns:
            tuple: Predicted class name and confidence score
        """
        # Preprocess the image
        input_tensor = self.transform(image).unsqueeze(0)
        
        # Disable gradient computation for inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            class_name = self.classes[predicted_class.item()]
            confidence_score = confidence.item() * 100
            
            return class_name, confidence_score

# Initialize the classifier
classifier = BloodCellClassifier(MODEL_PATH, CLASSES)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Open the image
        image = Image.open(file.stream).convert('RGB')
        
        # Predict the class
        predicted_class, confidence = classifier.predict_class(image)
        
        return jsonify({
            'class': predicted_class,
            'confidence': f'{confidence:.2f}%'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
