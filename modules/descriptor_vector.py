import numpy as np
from skimage import feature
from skimage.color import rgb2gray

# Transform a [[n_images, h, l, 3]] shaped array with h, l constant
# into a [[n_images, k]] shaped array where k is a constant

def hog(images, *args, **kwargs):
	return np.array([ feature.hog(img, block_norm=kwargs.pop('block_norm', 'L2'), **kwargs )
										for img in images ])

def flatten(images):
	return np.array([ img.ravel() for img in images ])

def gray_flatten(images):
	return np.array([ rgb2gray(img).ravel() for img in images ])

def daisy(images):
	return np.array([ feature.daisy(rgb2gray(img)) for img in images ])
