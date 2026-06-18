from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import os
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'static/uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Function to load model safely
def safe_load_model(model_path):
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model at {model_path}: {e}")
        return None

# Load the models
models = {
    "Alzheimers": {
        "model": safe_load_model(r"C:\Users\lumin\Documents\Demo\DiseasePrediction\model\dementia_classification_model.h5"),
        "class_labels": ['Non Demented', 'Mild Dementia', 'Moderate Dementia', 'Very Mild Dementia'],
        "input_shape": (128, 128)
    },
    "Brain_tumor": {
        "model": safe_load_model(r"C:\Users\lumin\Documents\Demo\DiseasePrediction\model\BrainTumor (2).h5"),
        "class_labels": ['glioma', 'meningioma', 'notumor', 'pituitary'],
        "input_shape": (224, 224)
    },
    "Diabetic": {
        "model": safe_load_model(r"C:\Users\lumin\Documents\Demo\DiseasePrediction\model\Diabetic(1).h5"),
        "class_labels": ['DR', 'No_DR'],
        "input_shape": (224, 224)
    },
    "Kidney": {
        "model": safe_load_model(r"C:\Users\lumin\Documents\Demo\DiseasePrediction\model\KidneyCTscan(3).h5"),
        "class_labels": ['Cyst', 'Normal', 'Stone', 'Tumor'],
        "input_shape": (224, 224)
    },
    "Respiratory": {
        "model": safe_load_model(r"C:\Users\lumin\Documents\Demo\DiseasePrediction\model\Respiratory (2).h5"),
        "class_labels": ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia'],
        "input_shape": (128, 128)
    }
}

# Preprocess the image for the selected model
def preprocess_image(image_path, model_key):
    model_info = models.get(model_key)
    input_shape = model_info["input_shape"]

    if model_key == "Alzheimers":
        img = Image.open(image_path)
        img = img.resize(input_shape)
        img = np.array(img).reshape(1, input_shape[0], input_shape[1], 3)
    else:
        img = tf.io.read_file(image_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, size=input_shape)
        img = img / 255.0
        img = tf.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        model_key = request.form.get('model_key')
        imagefile = request.files['imagefile']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
        imagefile.save(image_path)

        model_info = models.get(model_key)
        if not model_info or not model_info["model"]:
            raise ValueError(f"Model for {model_key} could not be loaded. Please check the model file.")

        # Preprocess and predict
        img_array = preprocess_image(image_path, model_key)
        model = model_info["model"]
        class_labels = model_info["class_labels"]

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        confidence = predictions[0][class_index] * 100
        prediction = f"{confidence:.2f}% Confidence: {class_labels[class_index]}"

        image_url = f"/{app.config['UPLOAD_FOLDER']}/{imagefile.filename}"
        return render_template('index.html', prediction=prediction, image_url=image_url)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', error=str(e))

@app.route('/static/uploads/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(port=5002, debug=True)
