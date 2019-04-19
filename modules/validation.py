import numpy as np
from .utils import area_rate
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

COVER_RATE = 0.5

def get_results_from_scores(scores, test_labels, limit_score, **kwargs):
	"""Compute results true/false positive/negative from scores and labels"""
	y_true = test_labels[:,5] == 1
	y_pred = scores > limit_score

	precision, recall, _ = precision_recall_curve(y_true, y_pred)
	results = {
		'avg_precision': average_precision_score(y_true, y_pred),
		'precision': precision,
		'recall': recall,
		'f1-score': f1_score(y_true, y_pred),
		'roc_auc_score': roc_auc_score(y_true, y_pred),
	}

	# print(f"Average precision-recall score: {results['avg_precision']:0.2f}")
	title = kwargs.get('title', 'Precision-Recall curve')

	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title(f"{title}: AP={results['avg_precision']:0.2f}")
	plt.show()

	return results

def get_false_positives(predictions, labels, display_info=False):
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
		correct_face = any([ area_rate(label[1:5], prediction[1:5]) > COVER_RATE
										 for label in img_labels ])

		# Keep false positives
		# Change format, to become a label (score -> class = -1)
		if not correct_face:
			false_positives.append([ *prediction[0:5], -1])

	if display_info:
		print(f"{len(false_positives)} false positives / {len(predictions)} predictions")
		print(f"Taux de réussite : {100 - len(false_positives)/len(predictions)*100:.2f}%")

	return np.array(false_positives, dtype=int)

def rate_predictions(predictions, labels):
	# Take only real faces
	labels = labels[labels[:,5] == 1]
	img_ids = np.unique(labels[:,0])
	results = {
		'true_pos': 0,
		'false_pos': 0,
		'missing_predictions': 0, # = false_neg

		'no_predictions': len(predictions),
		'no_faces': len(labels),
		'no_images': len(img_ids),
	}

	# Check every labelled image
	for img_id in img_ids:
		# TODO Optimize from 2n² to 2n
		img_labels = labels[labels[:,0] == img_id]
		img_predictions = predictions[predictions[:,0] == img_id]

		# Count missing predictions
		label_pred_diff = len(img_labels) - len(img_predictions)
		if label_pred_diff > 0:
			results['missing_predictions'] += label_pred_diff

		for prediction in img_predictions:
			# Check if the prediction covers a true face
			correct_face = any([ area_rate(label[1:5], prediction[1:5]) > COVER_RATE
											 for label in img_labels ])
			results['true_pos' if correct_face else 'false_pos'] += 1

	return results

def pr_curve(labels, predictions, scores=None):
	labels = labels[labels[:,5] == 1]
	img_ids = np.unique(labels[:,0])
	# predictions = predictions[np.argsort(scores)[::-1]]

	precision, recall = [], []
	results = {
		'true_pos': 0,
		'false_pos': 0,
		'false_neg': 0,
	}
	def update_pr():
		if results['true_pos'] + results['false_pos'] == 0:
			precision.append(0)
		else:
			precision.append(results['true_pos'] / (results['true_pos'] + results['false_pos']))
		if results['true_pos'] + results['false_neg'] == 0:
			recall.append(0)
		else:
			recall.append(results['true_pos'] / (results['true_pos'] + results['false_neg']))


	# Check every labelled image
	for img_id in img_ids:
		# TODO Optimize from 2n² to 2n
		img_labels = labels[labels[:,0] == img_id]
		img_predictions = predictions[predictions[:,0] == img_id]

		# Add missing predictions
		label_pred_diff = len(img_labels) - len(img_predictions)
		if label_pred_diff > 0:
			for _ in range(label_pred_diff):
				results['false_neg'] += 1
				update_pr()

		# Check positive predictions
		for prediction in img_predictions:
			# Check if the prediction covers a true face
			correct_face = any([ area_rate(label[1:5], prediction[1:5]) > COVER_RATE
											 for label in img_labels ])
			results['true_pos' if correct_face else 'false_pos'] += 1
			update_pr()

	plt.step(recall, precision, color='b', alpha=0.2, where='post')
	plt.fill_between(recall, precision, alpha=0.2, color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.05])
	# plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
	# auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])