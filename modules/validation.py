import numpy as np
from .utils import area_rate

COVER_AREA = 0.5

def get_results_from_scores(scores, test_labels, limit_score, display_info=False):
	"""Compute results true/false positive/negative from scores and labels"""
	results = {
		'true_pos': 0,
		'true_neg': 0,
		'false_pos': 0,
		'false_neg': 0,
	}

	# Compute results
	for index, score in enumerate(scores):
		pred_pos = score > limit_score
		test_pos = test_labels[index,5] == 1

		if pred_pos:
			results['true_pos' if test_pos else 'false_pos'] += 1
		else:
			results['false_neg' if test_pos else 'true_neg'] += 1

	# Compute precision and recall
	precision = results['true_pos'] / (results['true_pos'] + results['false_pos'])
	recall = results['true_pos'] / (results['true_pos'] + results['false_neg'])
	results['f-score'] = 2 * (precision * recall) / (precision + recall)
	results['precision'] = precision
	results['recall'] = recall

	if display_info:
		print("Prediction results:", results)

	return results

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
