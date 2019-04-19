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
	sample_size = kwargs.pop('sample_size', DEFAULT_SAMPLE_SIZE)
	params = {
		'feature_type': kwargs.pop('feature_type', DEFAULT_HAAR_FEATURE_SET[0]),
		'feature_coord': kwargs.pop('feature_coord', DEFAULT_HAAR_FEATURE_SET[1]),
	}
	first_feat = compute_haar(integral_image(compress_image(images[0], sample_size)), **params)
	results = np.array((len(images)), n_features)
	for i, img in images:
		if index == 0:
			results[index] = first_feat
		else:
			results[index] = compute_haar(integral_image(compress_image(img, sample_size)), **params)
	return results

def compute_haar(int_img, **kwargs):
	return feature.haar_like_feature(int_img, 0, 0, *int_img.shape[::-1], **kwargs)

def daisy(images, **kwargs):
	"""Downsample and compute daisy for each image"""
	# Get params
	kwargs = kwargs.copy()
	sample_size = kwargs.pop('sample_size', DEFAULT_SAMPLE_SIZE)

	first = feature.daisy(compress_image(images[0], sample_size), **kwargs)
	vectors = np.empty((len(images), *first.shape))

	# Compute each vector
	for index, img in enumerate(images):
		if index == 0:
			vectors[index] = first
		else:
			vectors[index] = feature.daisy(compress_image(img, sample_size), **kwargs)

	return vectors

