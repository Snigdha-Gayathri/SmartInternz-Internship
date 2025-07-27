from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import random
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = load_model('healthy_vs_rotten.h5')

CLASS_NAMES = ['biodegradable', 'recyclable', 'trash']
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_and_predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return 'No file uploaded', 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.

    prediction = model.predict(img_tensor)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]

    return render_template('portfolio-details.html', 
                           filename=filename, 
                           label=predicted_class)

@app.route('/random_predict', methods=['POST'])
def random_predict():
    folder_path = os.path.join(app.config['UPLOAD_FOLDER'])
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        return "No images in uploads folder", 404

    filename = random.choice(image_files)
    filepath = os.path.join(folder_path, filename)

    img = image.load_img(filepath, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.

    prediction = model.predict(img_tensor)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]

    return render_template('ipython.html', filename=filename, label=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
