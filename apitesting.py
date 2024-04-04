from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import tensorflow as tf


app = Flask(__name__)

# Load the model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Define image preprocessing function
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = img[np.newaxis, ...]
    return img.astype(np.float32)

# Define endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Save the file temporarily
        temp_path = 'temp.jpg'
        file.save(temp_path)
        
        # Preprocess the image
        input_data = preprocess_image(temp_path)
        
        # Make prediction
        interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        
        # Get predicted class
        predicted_class_index = np.argmax(output)
        predicted_class = categories[predicted_class_index]
        
        # Delete the temporary file
        os.remove(temp_path)
        
        return jsonify({'result': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
