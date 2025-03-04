from flask import Flask, render_template, request, jsonify
import torch
import timm
from torchvision import transforms
from PIL import Image
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=8)
        model_path = 'student_model.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise
    return model, device

student_model, device = load_model()

class_names = ['basophil', 'eosinophil', 'erythroblast', 'ig', 'lymphocyte', 'monocyte', 'neutrophil', 'platelet']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        try:
            logger.info(f"Processing file: {file.filename}")
            image = Image.open(file).convert('RGB')
            input_tensor = transform(image)
            input_batch = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = student_model(input_batch)
                probabilities = torch.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1)
                confidence = probabilities[0][predicted_class].item()

            predicted_label = class_names[predicted_class.item()]
            logger.info(f"Prediction: {predicted_label}, Confidence: {confidence:.2%}")
            
            return jsonify({
                'prediction': predicted_label,
                'confidence': f"{confidence:.2%}",
                'image_path': None
            })

        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)