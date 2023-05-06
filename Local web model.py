from io import BytesIO
from PIL import Image
import numpy as np
from pywebio import start_server
from pywebio.input import file_upload
from pywebio.output import put_image, put_text, put_buttons
import pickle
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import ResNet50


print(os.getcwd())
print(os.listdir())
# Load the saved SVM model
# load the model from file
with open('C:/Users/johns/OneDrive/University work/Final Year/Final Year Project/Python/SVM 2/RGB_SVM_model2.sav', 'rb') as f:
    svm_model = pickle.load(f)


# Define image size
image_size = (512, 512)
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=image_size+(3,))
# Function to preprocess the image
def preprocess_image(image):
    x = np.array(image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Function to classify the image
def classify_image(image_bytes):
    # Load the image
    image = Image.open(BytesIO(image_bytes)).resize(image_size)

    # Preprocess the image
    x = preprocess_image(image)

    # Load the saved SVM model
    with open('C:/Users/johns/OneDrive/University work/Final Year/Final Year Project/Python/SVM 2/RGB_SVM_model2.sav', 'rb') as f:
        svm_model = pickle.load(f)

    # Extract features from image
    features = resnet_model.predict(x)

    # Flatten the features array
    features = features.reshape(features.shape[0], -1)

    # Make prediction using SVM model
    prediction = svm_model.predict(features)[0]

    # Get the confidence score
    decision_values = svm_model.decision_function(features)
    confidence = abs(decision_values[0])

    # Determine the predicted class label based on the decision values
    if decision_values[0] >= 0:
        predicted_class = 'Glaucoma'
    else:
        predicted_class = 'Not Glaucoma'
        
    return predicted_class, confidence
    
# Define the web interface
def app():
    put_text('Upload an image to classify')
    image_bytes = file_upload('Select an image file')

    if image_bytes is not None:
        prediction, confidence = classify_image(image_bytes['content'])
        filename = image_bytes["filename"]
        put_text('Prediction: %s (%.2f%% confidence)' % (prediction, confidence*100))


        image = Image.open(BytesIO(image_bytes['content']))
        image.thumbnail((512, 512))
        put_image(image)

        put_buttons(["go back"], onclick=lambda _: app())
        
# Start the server
start_server(app, port=8888)