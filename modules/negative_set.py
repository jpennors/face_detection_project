import numpy as np
from random import randint, randrange
from .utils import area_rate


def check_no_overlap(labels, box):
	"""Check if the generated box doesn't overlap with any faces"""
	return all([ area_rate(box, label[1:5]) < 1/2 for label in labels ])

def get_box_parameters(labels, display_info=False):
	"""Compute the average box parameters"""
	h_mean = np.mean(labels[:,3])
	l_mean = np.mean(labels[:,4])
	ratio = np.mean(labels[:,3] / labels[:,4])

	if display_info:
		print("Le ration entre la hauteur et la largeur est de", ratio)
		print("La hauteur moyenne est de", h_mean)
		print("La largeur moyenne est de", l_mean)

	return ratio, int(h_mean), int(l_mean)

def generate_box(img, box_height, box_width):
	"""Generate a random box (h, l, x, y)"""
	#h = randint(50,150)
	h = box_height
	# l = int(h/ratio)
	l = box_width
	x = randrange(img.shape[1] - l)
	y = randrange(img.shape[0] - h)

	return (h,l,x,y)

def generate_negative_set(images, labels, set_size=300):
	"""
	@brief      Generate a set of negative labels from images 
	
	@param      images  The images
	@param      labels  The labels
	
	@return     The set of negative examples with class -1
	"""
	box_ratio, box_height, box_width = get_box_parameters(labels)
	n_images = len(images)

	neg_set = []
	while len(neg_set) < set_size :
		# Generate a fake box in a random image
		img_index = randrange(n_images)
		img_id = img_index + 1
		box = generate_box(images[img_index], box_height, box_width)

		# Check if it doesn't overlap with true faces
		img_labels = labels[labels[:,0] == img_id]
		if check_no_overlap(img_labels, box):
			neg_set.append([ img_id, *box, -1 ])

	return np.array(neg_set, dtype=int)

