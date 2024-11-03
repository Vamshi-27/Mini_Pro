from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

def extract_features(directory):
    model = VGG16()
    model.layers.pop()  # Remove the last layer (softmax)
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)  # Create a new model with the modified structure
    features = {}
    
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        try:
            image = load_img(img_path, target_size=(224, 224))  # Load the image and resize
            image = img_to_array(image)  # Convert to array
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            image = image / 255.0  # Normalize the image
            feature = model.predict(image, verbose=0)  # Get features
            image_id = img_name.split('.')[0]
            features[image_id] = feature
        except Exception as e:
            print(f"Could not process image {img_name}: {e}")
    
    return features