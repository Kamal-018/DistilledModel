from flask import Flask, render_template, request, jsonify
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load and set up the model
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=8)
    model_path = os.path.join('student_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

student_model, device = load_model()

# Define class names
class_names = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

# Define preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        try:
            # Save and process the image
            image_path = os.path.join('static', 'uploads', file.filename)
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            file.save(image_path)
            
            # Open and preprocess the image
            image = Image.open(image_path).convert('RGB')
            input_tensor = transform(image)
            input_batch = input_tensor.unsqueeze(0).to(device)

            # Make prediction
            with torch.no_grad():
                output = student_model(input_batch)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = probabilities[0][predicted_class].item()

            # Get the predicted class name
            predicted_label = class_names[predicted_class.item()]
            
            # Return JSON response for AJAX
            return jsonify({
                'prediction': predicted_label,
                'confidence': f"{confidence:.2%}",
                'image_path': f"/static/uploads/{file.filename}"
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)