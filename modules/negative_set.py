import numpy as np
from random import randint, randrange
from .utils import area_rate, get_shape_stats


def check_no_overlap(labels, box):
	"""Check if the generated box doesn't overlap with any faces"""
	return all([ area_rate(box, label[1:5]) < 1/2 for label in labels ])

def get_box_parameters(labels):
	"""Compute the average box parameters"""
	return get_shape_stats(labels[:,(3,4)])

def generate_box(img, box_ratio):
	"""Generate a random box (x, y, h, l)"""
	img_h, img_l = img.shape[:2]

	if int(img_h / box_ratio) < img_l:
		h = randint(int(40*box_ratio)+1, img_h - 1)
	else :
		h = randint(int(40*box_ratio)+1, int(img_l*box_ratio) - 1)

	l = int(h/box_ratio)
	x = randint(0, img_h - h)
	y = randint(0, img_l - l)

	if img_h < x+h or img_l < y+l:
		print(img_l - l, img_h - h)
		print(img_l, img_h, y+h, x+l)
	
	return x, y, h, l

def generate_negative_set(images, labels, set_size=300):
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

	return np.array(neg_set, dtype=int)

