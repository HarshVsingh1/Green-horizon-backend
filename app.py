import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

app = Flask(__name__)

model = ResNet50(weights='imagenet') 

@app.route('/predict', methods=['POST'])
def predict():
  
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Process the image
    img = image.load_img(file, target_size=(224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array) 

    preds = model.predict(img_array)
    decoded_preds = decode_predictions(preds, top=3)[0] 

    results = [{'class': cls, 'probability': float(prob)} for (_, cls, prob) in decoded_preds]
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=3000)
