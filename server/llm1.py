from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.applications.inception_v3 import InceptionV3, preprocess_input 
from PIL import Image
import numpy as np
from flask_cors import CORS
import json
import io

app = Flask(__name__)
CORS(app,supports_credentials=True)

# Load the pre-trained model
model = load_model("C:/Users/srial/OneDrive/Documents/psschool/clientf/server/indianamodelTrue.keras")

# Load the word-to-index and index-to-word mappings
with open('wordtoix.json', 'r') as f:
    wordtoix = json.load(f)

with open('ixtoword.json', 'r') as f:
    ixtoword = json.load(f)

base_model = InceptionV3(weights = 'imagenet') 
model1 = Model(base_model.input, base_model.layers[-2].output)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((299, 299))  # Resize to the size expected by the model
    image=image.convert('RGB')
    image = np.array(image) /255.0
    #image = preprocess_input(image)  # Preprocess the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print("image is returned after preprocessing....")
    return image


def encode(image): 
    image = preprocess_image(image) 
    print("image is sending for prediction...")
    vec = model1.predict(image) 
    vec = np.reshape(vec, (vec.shape[1])) 
    return vec 

# Function to generate caption
def generate_caption(image):
    
    print("for caption generation....")
    # Placeholder: Start the caption generation with "startseq"
    caption = 'startseq'
    
    # Maximum length of caption
    max_length = 40

    # Generate the caption
    for i in range(max_length):
        # Convert caption words to indices
        sequence = [wordtoix[w] for w in caption.split() if w in wordtoix]
        sequence = np.pad(sequence, (0, max_length - len(sequence)), mode='constant')
        sequence = np.array([sequence])

        # Predict the next word
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[str(yhat)]
        
        # Append the predicted word to the caption
        caption += ' ' + word

        # Stop if the end sequence token is predicted
        if word == 'endseq':
            break

    # Remove the start and end sequence tokens from the caption
    caption = caption.split()
    caption = caption[1:-1]
    caption = ' '.join(caption)

    return caption

@app.route('/upload', methods=['POST'])
def upload_image():
    print("image uploaded.....")
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Open the image file
    image = Image.open(io.BytesIO(file.read()))
    print("image is converted into file...")
    # Generate the caption
    image_features = encode(image).reshape(1,2048)
    print("image features......")
    caption = generate_caption( image_features)
    
    print(caption)
    
    return jsonify({'caption': caption})

if __name__ == '__main__':
    app.run(debug=True,port=5001)

