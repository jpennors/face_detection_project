import numpy as np
from skimage.io import imread
from skimage.util import img_as_float
from skimage.transform import resize
import os
import pickle

LABEL_PATH = 'data/label.txt'
TRAIN_PATH = 'data/train/'
TEST_PATH  = 'data/test/'
PREDICTION_PATH = 'detection.txt'
MODEL_PATH = 'model.pickle'

# Labels

def load_labels(path=LABEL_PATH, limit=None):
	"""Labels are of the following format: [[image_id, x, y, h, l, class]]"""
	labels = np.loadtxt(path, dtype=int)
	if limit:
		labels = labels[labels[:,0] <= limit]

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

def extract_boxes(images, labels):
	"""Extract all the labels boxes from the images"""
	boxes = []
	current_idx = None
	current_img = None
	for img_id, x, y, h, l, _ in labels:
		# Get image if needed
		if img_id != current_idx:
			current_idx = img_id
			# current_img = img_as_float(imread(f"./train/{str(img_id).zfill(4)}.jpg"))
			current_img = images[int(img_id)-1]

		# Extract box
		i = current_img[int(x):int(x+h), int(y):int(y+l)]
		if not i.shape:
			import pdb; pdb.set_trace()
		boxes.append(i)

	return np.array(boxes)

def compress_images(images, size):
	return np.array([ resize(img, size, mode='constant', anti_aliasing=True) for img in images ])

def compress_image(img,size):
	return resize(img, size, mode='constant', anti_aliasing=True)

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

# Training - Validation sets

def train_valid_sets(n_images, labels, train_rate=0.75):
	"""Create training and validation sets"""
	rnd_indexes = np.random.permutation(n_images)
	tv_limit = int(n_images * train_rate)
	label_corrected_ids = labels[:,0] - 1

	train_labels = labels[np.isin(label_corrected_ids, rnd_indexes[:tv_limit])]
	valid_labels = labels[np.isin(label_corrected_ids, rnd_indexes[tv_limit:])]
	return train_labels, valid_labels

