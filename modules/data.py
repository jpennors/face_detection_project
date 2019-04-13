import numpy as np
from skimage.io import imread
from skimage.util import img_as_float
from skimage.transform import resize
import os

LABEL_PATH = 'data/label.txt'
TRAIN_PATH = 'data/train/'
TEST_PATH  = 'data/test/'

def load_labels(path=LABEL_PATH):
	"""Labels are of the following form: [[image_id, x, y, h, l]]"""
	return np.loadtxt(path, dtype=int)

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
	for idx, x, y, h, l in labels:
			# Get image if needed
			if idx != current_idx:
					current_idx = idx
					# current_img = img_as_float(imread(f"./train/{str(idx).zfill(4)}.jpg"))
					current_img = images[idx-1]
			# Extract face
			faces.append(current_img[x:x+h, y:y+l])

	return np.array(faces)

def compress(images, size):
	return np.array([ resize(img, size, mode='constant', anti_aliasing=True) for img in images ])
