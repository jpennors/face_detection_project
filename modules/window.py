import numpy as np
from skimage.transform import resize
from .utils import area_rate, tqdm
from math import ceil

DEFAULT_DOWNSCALE_STEP = 50 # Downscale by 50px
DEFAULT_SLIDE_STEP = (60, 50)

def extract_boxes(images, labels, box_size):
	"""
	Extract all the labels boxes from the images in the same shape
	"""
	boxes = []
	current_idx = None
	current_img = None
	for img_id, x, y, h, l, _ in labels:
		# Get image if needed
		if img_id != current_idx:
			current_idx = img_id
			# current_img = img_as_float(imread(f"./train/{str(img_id).zfill(4)}.jpg"))
			current_img = images[img_id - 1]

		# Extract box
		box = compress_image(current_img[x:x+h, y:y+l], box_size)
		boxes.append(box)

	return np.array(boxes)

def compress_image(img, size):
	return resize(img, size, mode='constant', anti_aliasing=True)

def compute_no_windows(img, box_size, slide_step, downscale_step):
	"""Compute the number of sliding windows given for an image"""
	box_h, box_l = box_size
	slide_h, slide_l = slide_step
	min_height,	min_width = box_size

	h, l = img.shape[:2]
	ratio = h / l
	no_windows = 0
	no_images = 0
	while h > min_height and l > min_width:
		n_h = ceil((h - box_h) / slide_h)
		n_l = ceil((l - box_l) / slide_l)
		no_windows += n_h * n_l
		no_images += 1
		h = int(h - downscale_step)
		l = int(h / ratio)

	return no_windows


def downscale_image(img, step, min_height=100, min_width=100):
	"""
	@brief      Generate resized image by decreasing the height of 'step' px

	@param      img         The image to downscale
	@param      step        The number of pixels or percent used to decreased the height of the picture
	@param      min_height  The minimum height to generate
	@param      min_width   The minimum width to generate

	@return     Generator of downscaled images
	"""
	assert 0 <= step
	yield img
	h, l = img.shape[:2]

	if step == 0:
		# Yield only the original image
		return None
	elif step < 1:
		# Downscale by percent
		r_min = int(min(min_height / h, min_width / l) * 100)
		step = int(step * 100)
		for r in reversed(range(r_min, 100, step)):
			if r == 100:
				continue
			size = int(h * r / 100), int(l * r / 100)
			yield compress_image(img, size)
	else:
		# Downscale by pixels
		step = int(step)
		ratio = h / l
		no_images = 0
		while h - step > min_height and int((h - step)/ratio) > min_width:
			h = int(h - step)
			l = int(h / ratio)
			no_images += 1
			yield compress_image(img, (h, l))

def sliding_windows(img, box_size, step=None, downscale_step=None):
	"""
	@brief		Slide accross an image and pick window regions

	@param		img							The image to slide accross
	@param		box_size				The size of the window
	@param		step						The step by which to slide the window in x and y
	@param		downscale_step	The step by which to downscale the image

	@return		Set of windows of the following shape [[x, y, window]]
	"""
	# Clean params
	if step is None:
		step = DEFAULT_SLIDE_STEP
	if downscale_step is None:
		downscale_step = DEFAULT_DOWNSCALE_STEP
	if type(step) in (int, float):
		step = (step, step)
	if len(step) != 2:
		raise ValueError("There must be two values for the step")

	# Get params
	ini_img_h, ini_img_l = img.shape[:2]
	step_h, step_l = step
	box_h, box_l = box_size
	min_h, min_l = box_size
	if step_h >= box_h or step_l >= box_l:
		raise ValueError("The steps must be less than the box size")

	# Allocate space
	n_results = compute_no_windows(img, box_size, step, downscale_step)
	coordinates = np.empty((n_results, 4), dtype=int)
	windows = np.empty((n_results, *box_size, 3))

	index = 0
	for scaled_img in downscale_image(img, downscale_step, min_h, min_l):
		img_h, img_l = scaled_img.shape[:2]
		r_h = img_h / ini_img_h
		r_l = img_l / ini_img_l

		for x in range(0, img_h, step_h):
			for y in range(0, img_l, step_l):

				if x + box_h < img_h and y + box_l < img_l:
					# Window is at box_size for classification
					# but coordinates is not for detections
					window = scaled_img[x:x+box_h, y:y+box_l]
					coordinates[index] = [ x/r_h, y/r_l, box_h/r_h, box_l/r_l ]
					windows[index] = window
					index += 1

	return coordinates, windows

def multiple_sliding_windows(images, *args, **kwargs):
	only = kwargs.pop('only', None)
	indexes = []
	coordinates = []
	windows = []

	count = len(images) if only is None else len(only)
	for index, img in enumerate(tqdm(images, desc="Sliding windows", total=count)):
		if only is not None and index not in only:
			continue
		coord, window = sliding_windows(img, *args, **kwargs)
		indexes.extend([ index ] * len(coord))
		coordinates.extend(coord)
		windows.extend(window)

	# Beware, not optimized
	return np.array(indexes, dtype=int), np.array(coordinates, dtype=int), np.array(windows)

def filter_window_results(ids, coordinates, predictions, limit):
	"""Retrieve faces positive predictions from all predicitions""" 
	# Keep predictions where face recognition class is higher than limit
	positive_indices = np.where(predictions[:] > limit)

	single_index = not hasattr(ids, '__len__')
	if not single_index:
		ids = ids[positive_indices]
	positive_predictions = predictions[positive_indices]
	positive_coordinates = coordinates[positive_indices]

	# Sort remaining predictions by decreasing order
	sorted_indices = np.argsort(positive_predictions[:])[::-1]

	# Remove some boxes based on area rate
	face_boxes = []
	removed_indices = []
	for i in sorted_indices:
		if i not in removed_indices:
			coord = positive_coordinates[i]
			score = positive_predictions[i]
			img_id = ids if single_index else ids[i]
			face_boxes.append([ img_id, *coord, score ])
			for j in sorted_indices:
				if i != j and area_rate(positive_coordinates[i], positive_coordinates[j]) > 1/2 :
					removed_indices.append(j)

	return np.array(face_boxes)
