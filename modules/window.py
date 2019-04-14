from skimage.transform import resize

DEFAULT_STEP = 50 # Downscale by 50px

def down_image_pyramid(img, step=DEFAULT_STEP, min_height=100, min_width=100):
	"""
	@brief      Generate resized image by decreasing the height of 'step' px
	
	@param      img         The image to downscale
	@param      step        The number of pixels or percent used to decreased the height of the picture
	@param      min_height  The minimum height to generate
	@param      min_width   The minimum width to generate
	
	@return     Generator of downscaled images
	"""
	assert 0 < step
	yield img
	h, l = img.shape[:2]

	if step < 1:
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




