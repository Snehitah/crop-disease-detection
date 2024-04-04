from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from flask_cors import CORS 


app = Flask(__name__)
CORS(app)


# Load the model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Define image preprocessing function
def preprocess_image(image):
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = img[np.newaxis, ...]
    return img.astype(np.float32)

# Define endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'})
    
    image = request.files['image']
    
    if image.filename == '':
        return jsonify({'error': 'No selected image'})
    
    if image:
        # Preprocess the image
        input_data = preprocess_image(Image.open(image))
        
        # Make prediction
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        
        # Define categories
        categories = [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
            "Blueberry___healthy",
            "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
            "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
            "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
            "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
            "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
            "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
            "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
            "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
        ]
        
        # Get predicted class
        predicted_class_index = np.argmax(output)
        predicted_class = categories[predicted_class_index]
        
        return jsonify({'result': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
