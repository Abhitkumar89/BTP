from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import os
import cv2
import requests

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.models import load_model

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'static/uploaded_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('result.html')

loaded_model = load_model("cnn_lstm_model.keras")

img_size=224
categories = [
    'Normal - No signs of rheumatoid arthritis; joint structure appears healthy.',
    'Doubtful - Possible early-stage rheumatoid arthritis with minor joint space narrowing.',
    'Mild - Mild rheumatoid arthritis with small bone spurs and slight cartilage wear.',
    'Moderate - Moderate rheumatoid arthritis with noticeable joint space reduction and bone changes.',
    'Severe - Advanced rheumatoid arthritis with significant cartilage loss, bone damage, and joint deformity.'
]

# Load VGG16 without the top classification layers
vgg_base = VGG16(weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3))

for layer in vgg_base.layers:
    layer.trainable = False

feature_extractor = Model(inputs=vgg_base.input, outputs=vgg_base.output)

# Function to load and preprocess images
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("Image not found")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img / 255.0 

def predict_single_image(image_path):
    img = preprocess_image(image_path)
    img = np.expand_dims(img, axis=0)

    # Extract features using VGG16
    img_features = feature_extractor.predict(img, verbose=0)
    img_features = img_features.reshape(1, 7, -1)

    # Make a prediction
    prediction = loaded_model.predict(img_features)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Get class label
    class_label = categories[predicted_class]

    return class_label, confidence

@app.route('/result', methods=['GET', 'POST'])
def leaf():
    '''
    For rendering results on HTML GUI
    '''

    if request.method == 'POST':

        file = request.files['image']

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'img.jpg')
        file.save(file_path)

        class_label, confidence = predict_single_image(file_path)

        print(file_path)

        return render_template('result.html', predicted_label=class_label, img=file_path)

    return render_template('result.html')


if __name__ == "__main__":

    app.run(debug=True)
    