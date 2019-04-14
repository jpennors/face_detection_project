from skimage.transform import resize
import numpy as np

DEFAULT_DOWNSCALE_STEP = 50 # Downscale by 50px
DEFAULT_SLIDE_STEP = (20, 20)

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
			current_img = images[int(img_id)-1]

		# Extract box
		box = current_img[int(x):int(x+h), int(y):int(y+l)]
		box = compress_image(box, box_size)
		boxes.append(box)


	boxes =  np.array(boxes)
	if boxes.shape != 4:
		boxes

	return boxes

def compress_image(img, size):
	return resize(img, size, mode='constant', anti_aliasing=True)

def down_image_pyramid(img, step=DEFAULT_DOWNSCALE_STEP, min_height=100, min_width=100):
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
			yield resize(img, size, mode='constant', anti_aliasing=True)

	else:
		# Downscale by pixels
		step = int(step)
		ratio = h / l
		while h - step > min_height:
			h = int(h - step)
			l = int(h / ratio)
			yield resize(img, (h, l), mode='constant', anti_aliasing=True)

def sliding_windows(img, box_size, step=DEFAULT_SLIDE_STEP, downscale_step=DEFAULT_DOWNSCALE_STEP):
	"""
	@brief		Slide accros an image and pick window region

	@param		img		Image

	@return		Set of windows of the following shape [[x, y, window]]
	"""
	if type(step) in (int, float):
		step = (step, step)
	if len(step) != 2:
		raise ValueError("There must be two values for the step")

	step_h, step_l = step
	box_h, box_l = box_size
	if step_h >= box_h or step_l >= box_l:
		raise ValueError("The steps must be less than the box size")

	coordinates = []
	windows = []
	for scaled_img in down_image_pyramid(img, step=downscale_step):
		img_h, img_l = scaled_img.shape[:2]
		# TODO shift x, y ?

		for x in range(0, img_h, step_h):
			for y in range(0, img_l, step_l):

				# TODO Comment prendre le dernier ?
				if x + step_h + box_h < img_h and y + step_l + box_l < img_l:
					window = scaled_img[x:x+box_h, y:y+box_l]
					coordinates.append([x, y])
					windows.append(window)

	return np.array(coordinates), np.array(windows)
