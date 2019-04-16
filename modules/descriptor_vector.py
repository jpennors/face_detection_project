import numpy as np
from skimage import feature
from skimage.color import rgb2gray

# Transform a [[n_images, h, l, 3]] shaped array with h, l constant
# into a [[n_images, k]] shaped array where k is a constant

def hog(images):
	features = [ feature.hog(img, block_norm='L2') for img in images ]
	return np.array(features)

def flatten(images):
	return np.array([ img.flatten() for img in images ])

def gray_flatten(images):
	return np.array([ rgb2gray(img).flatten() for img in images ])
