import numpy as np
from keras.applications.vgg16 import VGG16
from keras.models import Model
from PIL import Image

def extract_features(image):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    image = Image.open(image).resize((224, 224))
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    return model.predict(image, verbose=0)