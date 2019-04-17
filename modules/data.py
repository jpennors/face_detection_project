import numpy as np
from skimage.io import imread, imsave
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
	if path == 'test':
		path = TEST_PATH
	labels = np.loadtxt(path, dtype=int)
	if limit:
		labels = labels[labels[:,0] <= limit]
	return np.append(labels, np.ones((labels.shape[0], 1), dtype=int), axis=1)

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

# Models

def load_model(path=MODEL_PATH):
	"""Load previous stored model"""
	return pickle.load(open(path, 'rb'))

def save_model(model, path=MODEL_PATH):
	"""Save model in a pickle file"""
	return pickle.dump(model, open(path, 'wb'))


# Training - Validation sets

def train_valid_sets(n_images, labels, train_rate=0.75):
	"""Create training and validation sets"""
	rnd_indexes = np.random.permutation(n_images)
	tv_limit = int(n_images * train_rate)
	label_corrected_ids = labels[:,0] - 1

	train_labels = labels[np.isin(label_corrected_ids, rnd_indexes[:tv_limit])]
	valid_labels = labels[np.isin(label_corrected_ids, rnd_indexes[tv_limit:])]
	return train_labels, valid_labels


# Save image
def save_img(path, img):

	imsave(path, img)

