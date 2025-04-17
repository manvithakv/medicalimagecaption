import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image as keras_image
from flask_cors import CORS
import numpy as np
import io

app = Flask(__name__)
CORS(app,supports_credentials=True)

# Load the pre-trained model
try:
    model = tf.keras.models.load_model("C:/Users/srial/OneDrive/Documents/psschool/clientf/server\model (1).h5")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", str(e))

def preprocess_image(image):
    try:
        # Preprocess the image
        img_array = keras_image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image to [0, 1] range
        print("Image preprocessing done....")
        return img_array
    except Exception as e:
        print("Error in preprocessing image:", str(e))
        raise e

@app.route('/')
def index():
    return "Flask server is running"

@app.route('/predict', methods=['POST'])
def predict():
    print("Prediction started.....")
    if 'image' not in request.files:
        print("No file part found in the request")
        return jsonify({'error': 'No file part'})

    file = request.files['image']

    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file'})

    if file:
        try:
            # Convert the file to a file-like object
            file_like = io.BytesIO(file.read())
            print("File found and converted to file-like object")
            
            # Preprocess the image
            img = keras_image.load_img(file_like, target_size=(224, 224))
            print("Image loaded successfully")
            
            img_array = preprocess_image(img)
            print("Preprocessed image array:", img_array)

            # Make prediction
            predicted_prob = model.predict(img_array)[0][0]
            print("Predicted probability:", predicted_prob)
            
            predicted_class = 'Normal' if predicted_prob > 0.5 else 'Effusion'
            print("Predicted class:", predicted_class)

            return jsonify({'prediction': predicted_class})
        except Exception as e:
            print("Error during prediction:", str(e))
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True,port=5002)


