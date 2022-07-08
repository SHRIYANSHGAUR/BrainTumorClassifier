# some utilities
import os
import numpy as np
from util import base64_to_pil

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

#tensorflow
import tensorflow as tf
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2



# Declare a flask app
app = Flask(__name__)

def get_ImageClassifierModel():
    model = tf.keras.models.load_model('BrainTumorClassification');

    return model




@app.route('/', methods=['GET'])
def index():
    '''
    Render the main page
    '''
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        img = img.resize((300, 300))
        # initialize model
        model = get_ImageClassifierModel()
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        class_names=['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

        pred_proba = "{:.2f}".format(100*np.max(score))    # NEW  Max probability
        type="{}".format(class_names[np.argmax(score)])
        # Serialize the result, you can add additional fields
        return jsonify(result=type, probability=pred_proba)
    return None


if __name__ == '__main__':
    
    app.run()
