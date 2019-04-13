import numpy as np
from skimage.io import imread
from skimage.util import img_as_float
import os

LABEL_PATH = 'data/label.txt'
TRAIN_PATH = 'data/train/'
TEST_PATH  = 'data/test/'

def load_labels(path=LABEL_PATH):
	"""Labels are of the following form: [[image_id, x, y, h, l]]"""
	return np.loadtxt(path)

def load_images(path=TRAIN_PATH, limit=None):
	images = []
	for index, img_file in enumerate(os.listdir(path)):
		if limit is not None and index >= limit: 
			break
		images.append(img_as_float(imread(path + img_file)))
	return np.array(images)
