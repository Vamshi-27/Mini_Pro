import os
import numpy as np
import pickle
from keras.applications.vgg16 import VGG16
from keras.models import Model
from PIL import Image

# Define the image directory and output pickle file path
images_path = 'D:/SIC/Flickr8k_Dataset'
output_path = 'data/features.pkl'

def extract_features(directory):
    model = VGG16()
    model.layers.pop()  # Remove the last layer (softmax layer)
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    features = {}
    for img_name in os.listdir(directory):
        img_path = os.path.join(directory, img_name)
        image = Image.open(img_path).resize((224, 224))
        image = np.array(image).astype('float32')
        image = np.expand_dims(image, axis=0)
        image = image / 255.0
        feature = model.predict(image, verbose=0)
        image_id = img_name.split('.')[0]
        features[image_id] = feature

    # Save features to a pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(features, f)
    
    print(f"Extracted features saved to {output_path}")

# Run the function to extract features
extract_features(images_path)
