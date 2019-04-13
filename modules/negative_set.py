import numpy as np
from random import randint, randrange

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

def check_overlap(labels, img_id, box):
	h2, l2, x2, y2 = box

	# Récupère les visages trouvés sur l'image
	faces = np.where(labels[:,0] == img_id)[0]
	for face_index in faces:
		_, x1, y1, h1, l1 = labels[face_index]
		
		if (((x1 < x2 + l2 and x1 + l1 > x2) or (x2 < x1 + l1 and x2 + l2 > x1))
		and ((y1 < y2 + h2 and y1 + h1 > y2) or (y2 < y1 + h1 and y2 + h2 > y1))):
			pass
		else: 
			xe_inter = max(x1, x2)
			ye_inter = max(y1, y2)

			ys_inter = min(y1+h1, y2+h2)
			xs_inter = min(x1+l1, x2+l2)

			aire_inter = (xs_inter-xe_inter) * (ys_inter-ye_inter)
			aire_union = ((h1*l1) + (h2*l2)) - aire_inter

			# The box overlap with another face
			if aire_inter / aire_union > 1/2 :
				return False
	# The box doesn't overlap with another face
	return True


def generate_negative_set(images, labels, set_size=300):
	"""
	@brief      Generate a set of negative labels from images 
	
	@param      images  The images
	@param      labels  The labels
	
	@return     The set of negative examples
	"""
	box_ratio, box_height, box_width = get_box_parameters(labels)
	n_images = len(images)

	neg_set = []
	while len(neg_set) < set_size :
		# Generate a fake box in a random image
		img_id = randrange(n_images)
		box = generate_box(images[img_id], box_height, box_width)

		# Check if it doesn't overlap with true faces
		if check_overlap(labels, img_id, box):
			neg_set.append([ img_id, *box ])

	return np.array(neg_set)

