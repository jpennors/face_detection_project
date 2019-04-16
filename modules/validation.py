import numpy as np
from .utils import area_rate

COVER_AREA = 0.5

def get_false_positives(predictions, labels, display_info=True):
	"""
	@brief Compute the error rate of predictions based on labels

	@param  predictions     Each prediction must be like this [img_id,x,y,h,l,score]
	@param  labels          A label is like this [img_id,x,y,h,l,class]
	"""
	# Take only real faces
	labels = labels[labels[:,5] == 1]
	false_positives = []

	for prediction in predictions:
		img_labels = labels[labels[:,0] == prediction[0]]

		# Check if the prediction covers a true face
		valid = False
		for img_label in img_labels:            
			if area_rate(img_label[1:5], prediction[1:5]) > COVER_AREA:
				valid = True
				break

		# Keep false positives
		# Change format, to become a label (score -> class = -1)
		if not valid:
			false_positives.append([ *prediction[0:5], -1])

	if display_info:
		print(f"{len(false_positives)} false positives / {len(predictions)} predictions")
		print(f"Taux de réussite : {100 - len(false_positives)/len(predictions)*100:.2f}%")

	return np.array(false_positives)