import numpy as np
from random import randint, randrange
from skimage.io import imsave
from .utils import area_rate, get_shape_stats
import os

MIN_BOX_HEIGHT = 60

def check_no_overlap(labels, box):
	"""Check if the generated box doesn't overlap with any faces"""
	return all([ area_rate(box, label[1:5]) < 1/3 for label in labels ])

def get_box_parameters(labels):
	"""Compute the average box parameters"""
	return get_shape_stats(labels[:,(3,4)])

def generate_box(img, box_ratio):
	"""Generate a random box (x, y, h, l)"""
	img_h, img_l = img.shape[:2]

	h = randint(MIN_BOX_HEIGHT, min(img_h - 1, img_l - 1))
	l = int(h/box_ratio)

	x = randrange(img_h - h)
	y = randrange(img_l - l)

	# if int(img_h / box_ratio) < img_l:
	# 	h = randint(40, img_h - 1)
	# else :
	# 	h = randint(40, int(img_l*box_ratio) - 1)

	# l = int(h/box_ratio)
	# x = randrange(img_h - h)
	# y = randrange(img_l - l)

	return x, y, h, l

def generate_negative_set(images, labels, set_size=300, save=False):
	"""
	@brief      Generate a set of negative labels from images 
	
	@param      images  The images
	@param      labels  The labels
	
	@return     The set of negative examples with class -1
	"""
	box_ratio = get_box_parameters(labels)[0]
	n_images = len(images)

	neg_set = []
	while len(neg_set) < set_size :
		# Generate a fake box in a random image
		img_index = randrange(n_images)
		img_id = img_index + 1
		box = generate_box(images[img_index], box_ratio)

		# Check if it doesn't overlap with true faces
		img_labels = labels[labels[:,0] == img_id]
		if check_no_overlap(img_labels, box):
			neg_set.append([ img_id, *box, -1 ])

	# Save images if true
	if save:
		save_negative_set(images, np.array(neg_set))

	return np.array(neg_set, dtype=int)


def save_negative_set(images, negatives):
	"""Function to save images from the negative set"""

	for i in range(len(negatives)):
		x = negatives[i][1]
		y = negatives[i][2]
		h = negatives[i][3]
		l = negatives[i][4]
		img = images[negatives[i][0]-1]
		negative_img = img[x:x+h,y:y+l]

		dir = "negative_set"

		if not os.path.exists(dir):
			os.makedirs(dir)

		imsave(f"{dir}/{i+1}-id-{negatives[i][0]}.png", negative_img)

