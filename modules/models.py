from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import numpy as np
from .negative_set import get_box_parameters
from .window import extract_boxes, sliding_windows, filter_window_results
from .validation import get_false_positives

LIMIT_SCORE = 0.5
BEST_MODEL = 'random_forest'

MODELS = {
	'svc': SVC,
	'linear_svc': LinearSVC,
	'decision_tree': DecisionTreeClassifier,
	'adaboost': AdaBoostClassifier,
	'random_forest': RandomForestClassifier,
}

DEFAULT_PARAMS = {
	'svc': {},
	'linear_svc': {},
	'decision_tree': {},
	'adaboost': {},
	'random_forest': {
		'n_estimators': 100,
	},
}

DECISION_METHODS = {
	'SVC': 										'decision_function',
	'LinearSVC': 							'decision_function',
	'AdaBoostClassifier' : 		'decision_function',
	'DecisionTreeClassifier': 'predict_proba',
	'RandomForestClassifier': 'predict_proba',
}

def create_model(class_name=BEST_MODEL, params=None):
	"""Easy constructor for models with default optimized params"""
	if class_name not in MODELS:
		raise NotImplementedError(
						f"Classifier {class_name} is not implemented in this function"
						f"\nYou can use: {', '.join(MODELS.keys())}")

	if not params:
		params = DEFAULT_PARAMS[class_name]

	return MODELS[class_name](**params)

def get_scores(clf, *args, **kwargs):
	"""Get the decision scores of a classifier for the +1 class"""
	name = DECISION_METHODS[clf.__class__.__name__]
	method = getattr(clf, name)
	scores = method(*args, **kwargs)

	# Only keep the positive, no problem here all are probalities
	# For DecisionTreeClassifier, RandomForestClassifier
	if scores.ndim == 2:
		positive_class_index = np.where(clf.classes_ == 1)[0][0]
		return scores[:, positive_class_index]

	assert scores.ndim == 1
	if type(clf) is LinearSVC:
		# score > 0 implies class +1
		return (scores / 2) + 0.5

	if type(clf) is AdaBoostClassifier:
		# score is around +1 and -1
		return scores - 0.5

	if type(clf) is SVC:
		kernel = clf.get_params().get('kernel')
		if kernel == 'linear':
			# score > 0 implies class +1
			return (scores / 2) + 0.5

		# TODO Learn more

	return scores


def train(clf, images, box_size, train_labels, vectorize, negatives=None, **kwargs):
	"""
	@brief      Train a classifier with the boxes labelled on the images
	
	@param      clf             The classifier instance
	@param      images          The images
	@param      labels          The labels
	@param      vectorize       The function used to vectorize the extracted boxes of images
	@param      vectorize_args  Arguments to be passed to the vectorize function
															in addition to the boxes
	@param      negatives       The negatives labels
	"""
	# Extract boxes of the images from the labels
	boxes = extract_boxes(images, train_labels, box_size)

	# Get the training set
	X = vectorize(boxes, *kwargs.get('vectorize_args', []))
	y = train_labels[:,5]

	# First training with only train_labels and random negatives
	print("First training...")
	clf.fit(X, y)

	# Beginning of the second training from the training images

	# TODO Predict all images ???
	# image_indexes = range(1, len(images)+1)
	train_images = images #[np.isin(image_indexes, train_labels[:,0])]
	predictions = predict(clf, train_images, box_size, vectorize)

	positive_false = get_false_positives(predictions, train_labels)
	train_labels = np.concatenate([train_labels, positive_false])
	print(f"Adding {len(positive_false)} false positives")

	# Extract new boxes of the images from the labels
	boxes = extract_boxes(images, train_labels, box_size)

	# Get the training set
	X = vectorize(boxes, *kwargs.get('vectorize_args', []))
	y = train_labels[:,5]

	# Finally, train again
	print("Second training...")
	clf.fit(X, y)

def accuracy(clf, images, box_size, labels, vectorize, negatives=None, **kwargs):
	boxes = extract_boxes(images, labels, box_size)

	# Get the training set
	X = vectorize(boxes, *kwargs.get('vectorize_args', []))
	y = labels[:,5]

	return clf.score(X, y)

def predict(clf, images, box_size, vectorize, **kwargs):
	"""
	@brief      Find faces on the images
	
	@param      clf        The trained classifier
	@param      images     The images
	@param      box_size   The box size
	@param      vectorize  The function used to vectorize the windows for the classifier
	
	@return     The covering boxes with their scores: [[ img_id, x, y, h, l, s ]]
	"""
	# Get params
	slide_step = kwargs.get('slide_step', (20, 20))
	downscale_step = kwargs.get('downscale_step', 0)

	results = []
	for index, image in enumerate(images):

		coordinates, windows = sliding_windows(image, box_size, slide_step, downscale_step)

		# Get the set and predict scores per class
		X = vectorize(windows, *kwargs.get('vectorize_args', []))
		scores = get_scores(clf, X)

		predictions = filter_window_results(index+1, coordinates, scores, LIMIT_SCORE)
		results.extend(predictions)
			
	return np.array(results)

def predict_and_validate(clf, images, box_size, test_labels, vectorize, **kwargs):
	"""
	@brief      Compute the accuracy of the trained classifier on labelled boxes
	
	@param      clf           The trained classifier
	@param      images        The images
	@param      box_size      The box size
	@param      test_labels   The test labels
	@param      vectorize     The function used to vectorize the windows for the classifier
	
	@return     The covering boxes with their scores: [[ img_id, x, y, h, l, s ]]
	"""

	boxes = extract_boxes(images, test_labels, box_size)

	X = vectorize(boxes, *kwargs.get('vectorize_args', []))
	scores = get_scores(clf, X)

	results = {
		'true_pos': 0,
		'true_neg': 0,
		'false_pos': 0,
		'false_neg': 0,
	}

	# Compute results
	for index, score in enumerate(scores):
		pred_pos = score > LIMIT_SCORE
		test_pos = test_labels[index,5] == 1

		if pred_pos:
			results['true_pos' if test_pos else 'false_pos'] += 1
		else:
			results['false_neg' if test_pos else 'true_neg'] += 1

	if kwargs.get('display_info', True):
		print("Prediction results:", results)
		print(f"Prediction accuracy for faces: {results['true_pos'] / sum(test_labels[:,5] == 1) * 100:.2f}%")

	return results
