import numpy as np
from skimage import feature
from skimage.color import rgb2gray
from skimage.transform import integral_image

# Transform a [[n_images, h, l, 3]] shaped array with h, l constant
# into a [[n_images, k]] shaped array where k is a constant

DEFAULT_HAAR_FEATURE = 'type-3-x'

def hog(images, *args, **kwargs):
	return np.array([ feature.hog(img, block_norm=kwargs.pop('block_norm', 'L2'), **kwargs )
										for img in images ])

def hog(images, *args, **kwargs):
	return np.array([ feature.hog(img, block_norm=kwargs.pop('block_norm', 'L2'), **kwargs )
										for img in images ])

def compute_integral_images(images):
	"""Compute integral images for a set of images"""
	return np.array([ integral_image(img) for img in images ])


def haar(int_images, feature_type=None, **kwargs):
	"""Extract the haar feature for the current image"""
	if feature_type is None:
		feature_type = DEFAULT_HAAR_FEATURE

	first_feat = compute_haar(int_img, feature_type=feature_type, **kwargs)
	results = np.array((len(int_images)), n_features)
	for index, int_img in enumerate(int_images):
		if index == 0:
			results[index] = first_feat
		else:
			results[index] = compute_haar(int_img, feature_type=feature_type, **kwargs)
	return results

def compute_haar(int_img, feature_type, **kwargs):
	return feature.haar_like_feature(int_img , 0, 0, *int_img.shape[:2], feature_type=feature_type, **kwargs)

def flatten(images):
	return np.array([ img.ravel() for img in images ])

def gray_flatten(images):
	return np.array([ rgb2gray(img).ravel() for img in images ])

def daisy(images):
	return np.array([ feature.daisy(rgb2gray(img)) for img in images ])
