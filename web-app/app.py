import os
from os import listdir
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import h5py
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from .model_splitter import split, combine

app = Flask(__name__)



def convert_image(image):
    # Convert to PIL Image
    pil_image = Image.open(image)

    # Resize image to same dimensions as data
    height = 800
    width = 1100
    resize_tuple = (width, height)
    pil_image = pil_image.resize(resize_tuple)

    # Convert PIL Image to Numpy Array
    img_array = tf.keras.preprocessing.image.img_to_array(pil_image)

    # Normalize pixel values
    img_array = img_array / 255.0

    # print(f"Size: {img_array.size}")
    # Convert to tensor
    tensor = tf.convert_to_tensor(img_array)
    # print(f"Shape {tf.shape(tensor)}")
    
    # Give batch size
    finished_tensor = tf.expand_dims(tensor, axis=0)
    # print(f"Shape {tf.shape(finished_tensor)}")

    return finished_tensor
    

def allowed_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg'))


# this code is to initially split the large model into subcomponents
'''
prefix = "model_weights_part"
fname_src = "./model/model.h5"
size_max = 90 * 1024**2  # maximum size allowed in bytes
fname_parts = split(fname_src, fname_dest_prefix=prefix, maxsize_per_file=size_max)
'''

# now we need to just get a list of the model part names

model_parts = [file for file in listdir("./model")]
combine(fname_in=model_parts, fname_out="model_weights.h5")

model = tf.keras.models.load_model('model_weights.h5')

@app.route('/')
def home():
    return render_template('index.html', output="None", prediction="N/A")

@app.route('/predict', methods=['POST'])
def predict():
    output = ""

    image = request.files['file']
    if not allowed_file(image.filename):
        output = "Invalid File Format"
        outputProb = "N/A"
    else:
        # Preprocessing
        preprocessed_img = convert_image(image)

        # Now Predict
        prediction = model.predict(preprocessed_img)
        
        
        if prediction.astype(float) > 0.5:
            isFraud = "Fraud!"
            prob = round(prediction.item(), 4) * 100
        else:
            isFraud = "Not Fraud!"
            prob = round((1 - prediction.item()), 4) * 100


        output = f"{isFraud}"
        outputProb = f"{prob}%"
    
    return render_template('index.html', output=output, prediction=outputProb) 


if __name__ == '__main__':
    app.run(debug=True)