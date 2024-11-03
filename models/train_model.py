import os
import pickle
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from PIL import Image
from define_model import define_model

# Paths
images_path = 'data/Flickr8k_Dataset'
captions_path = 'data/Flickr8k.token.txt'
features_path = 'data/features.pkl'
tokenizer_path = 'data/tokenizer.pkl'
model_path = 'models/model.h5'

# Load captions
def load_captions(filename):
    with open(filename, 'r') as file:
        captions = {}
        for line in file:
            tokens = line.strip().split()
            image_id, caption = tokens[0].split('.')[0], ' '.join(tokens[1:])
            captions.setdefault(image_id, []).append('startseq ' + caption + ' endseq')
    return captions

# Pre-process captions (lowercase and remove punctuation)
def preprocess_text(captions):
    import string
    table = str.maketrans('', '', string.punctuation)
    for key, caps in captions.items():
        caps[:] = [' '.join([w.translate(table).lower() for w in cap.split() if w.isalpha()]) for cap in caps]
    return captions

# Extract features using VGG16
def extract_features(directory):
    model = VGG16()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    features = {}
    for name in os.listdir(directory):
        img_path = os.path.join(directory, name)
        image = Image.open(img_path).resize((224, 224))
        image = np.expand_dims(np.array(image).astype('float32') / 255.0, axis=0)
        feature = model.predict(image, verbose=0)
        features[name.split('.')[0]] = feature
    return features

# Load or extract features
if not os.path.exists(features_path):
    features = extract_features(images_path)
    with open(features_path, 'wb') as f:
        pickle.dump(features, f)
else:
    with open(features_path, 'rb') as f:
        features = pickle.load(f)

# Load captions and preprocess
captions = load_captions(captions_path)
captions = preprocess_text(captions)

# Tokenize captions
tokenizer = Tokenizer()
all_captions = [cap for caps in captions.values() for cap in caps]
tokenizer.fit_on_texts(all_captions)
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(c.split()) for c in all_captions)

# Create sequences for training
def create_sequences(captions, photos, tokenizer, max_length):
    X1, X2, y = [], [], []
    for key, caps in captions.items():
        for cap in caps:
            seq = tokenizer.texts_to_sequences([cap])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = pad_sequences([seq[:i]], maxlen=max_length)[0], to_categorical([seq[i]], num_classes=vocab_size)[0]
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

X1, X2, y = create_sequences(captions, features, tokenizer, max_length)

# Define and train the model
model = define_model(vocab_size, max_length)
checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='loss', mode='min')
model.fit([X1, X2], y, epochs=20, callbacks=[checkpoint], verbose=1)
