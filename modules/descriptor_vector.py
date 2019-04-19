import numpy as np
from skimage import feature
from skimage.color import rgb2gray
from skimage.transform import integral_image
from .utils import compress_image

# Transform a [[n_images, h, l, 3]] shaped array with h, l constant
# into a [[n_images, k]] shaped array where k is a constant

DEFAULT_SAMPLE_SIZE = (40, 25)
DEFAULT_HAAR_FEATURE = 'type-3-x'
DEFAULT_HAAR_FEATURE_SET = feature.haar_like_feature_coord(
																*DEFAULT_SAMPLE_SIZE[::-1],
																feature_type=DEFAULT_HAAR_FEATURE)


def hog(images, **kwargs):
	"""Downsample and compute hog for each image"""
	# Get params
	kwargs = kwargs.copy()
	block_norm = kwargs.pop('block_norm', 'L2')
	sample_size = kwargs.pop('sample_size', DEFAULT_SAMPLE_SIZE)

	first = feature.hog(compress_image(images[0], sample_size), block_norm=block_norm, **kwargs)
	vectors = np.empty((len(images), *first.shape))

	# Compute each vector
	for index, img in enumerate(images):
		if index == 0:
			vectors[index] = first
		else:
			vectors[index] = feature.hog(compress_image(img, sample_size), block_norm=block_norm, **kwargs)

	return vectors

def haar(images, **kwargs):
	kwargs = kwargs.copy()
	params = {
		'feature_type': kwargs.get('feature_type', DEFAULT_HAAR_FEATURE[0]),
		'feature_coord': kwargs.get('feature_coord', DEFAULT_HAAR_FEATURE[1]),
	}
	
	first_feat = compute_haar(integral_image(compress_image(images[0]), **params))
	results = np.array((len(images)), n_features)
	for i, img in images:
		if index == 0:
			results[index] = first_feat
		else:
			results[index] = compute_haar(integral_image(compress_image(img), **kwargs))
	return results

def daisy(images, **kwargs):
	"""Downsample and compute daisy for each image"""
	# Get params
	kwargs = kwargs.copy()
	sample_size = kwargs.pop('sample_size', DEFAULT_SAMPLE_SIZE)

	first = feature.daisy(compress_image(images[0], sample_size), **kwargs)
	vectors = np.empty((len(images), *first.shape))
	print(vectors.shape)

	# Compute each vector
	for index, img in enumerate(images):
		if index == 0:
			vectors[index] = first
		else:
			vectors[index] = feature.daisy(compress_image(img, sample_size), **kwargs)

	return vectors
	

def compute_haar(int_img, feature_type, **kwargs):
	return feature.haar_like_feature(int_img , 0, 0, *int_img.shape[:2], **kwargs)


def compute_integral_images(images):
	"""Compute integral images for a set of images"""
	return np.array([ integral_image(img) for img in images ])


def haarold(int_images, feature_type=None, **kwargs):
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


def flatten(images):
	return np.array([ img.ravel() for img in images ])

def gray_flatten(images):
	return np.array([ rgb2gray(img).ravel() for img in images ])

def daisy(images):
	return np.array([ feature.daisy(rgb2gray(img)) for img in images ])
