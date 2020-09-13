"""
Caltech 256 API
Serves a pytorch caltech 256 model using Flask
Link to caltech 256 classes: https://gist.github.com/programming-datascience/ae995c2162e2b34373cfcb6146376fc2
Link to model download: https://www.dropbox.com/s/rjttmgcdp06i6k5/caltech256-script.zip?dl=0
"""

import io
import json
import torch

from torchvision import transforms
from PIL import Image
from flask import Flask, jsonify, request
from classes import caltech256Classes

app = Flask(__name__)

model = torch.jit.load("caltech256-script.zip", map_location="cpu") # Model download link is shown above too
model.eval() # Evaluation mode, IMPORTANT

def transform_image(image_bytes): 
    # We will recieve the image as bytes
    transform_image = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return transform_image(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    predicted_idx = y_hat.item()
    return caltech256Classes[predicted_idx]

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        image = request.files['file']
        img_bytes = image.read()
        class_name = get_prediction(img_bytes)
        return jsonify({
            "class_name":class_name
        })

if __name__ == '__main__':
    app.run(debug=True)