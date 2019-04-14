import numpy as np
from skimage.io import imread
from skimage.util import img_as_float
from skimage.transform import resize
import os
import pickle

LABEL_PATH = 'data/train/label.txt'
TRAIN_PATH = 'data/train/train/'
TEST_PATH  = 'data/test/'
PREDICTION_PATH = 'detection.txt'
MODEL_PATH = 'model.pickle'

# Labels

def load_labels(path=LABEL_PATH):
	"""Labels are of the following format: [[image_id, x, y, h, l, class]]"""
	labels = np.loadtxt(path, dtype=int)
	return np.append(labels, np.ones((labels.shape[0], 1)), axis=1)

def save_prediction(predictions, path=PREDICTION_PATH):
	"""Save predictions in the following format: [[image_id, x, y, h, l, score]]"""
	return np.savetxt(path, predictions)

# Images

def load_images(path=TRAIN_PATH, limit=None):
	images = []
	for index, img_file in enumerate(os.listdir(path)):
		if limit is not None and index >= limit: 
			break
		images.append(img_as_float(imread(path + img_file)))
	return np.array(images)

def extract_faces(images, labels):
	# Extract faces
	faces = []
	current_idx = None
	current_img = None
	for idx, x, y, h, l, _ in labels:
		# Get image if needed
		if idx != current_idx:
			current_idx = idx
			# current_img = img_as_float(imread(f"./train/{str(idx).zfill(4)}.jpg"))
		current_img = images[int(idx)-1]
		# Extract face
		faces.append(current_img[int(x):int(x+h), int(y):int(y+l)])

	return np.array(faces)

def compress(images, size):
	return np.array([ resize(img, size, mode='constant', anti_aliasing=True) for img in images ])

# Models

def load_model(path=MODEL_PATH):
	"""Load previous stored model"""
	return pickle.load(open(path, 'rb'))

def save_model(model, path=MODEL_PATH):
	"""Save model in a pickle file"""
	return pickle.dump(open(path, 'wb'), model)


# Test set

def load_test_images():
	return load_images(path=TEST_PATH)
