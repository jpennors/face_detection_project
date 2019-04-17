from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import numpy as np
from .negative_set import get_box_parameters
from .window import extract_boxes, sliding_windows, filter_window_results
from .validation import get_false_positives, get_results_from_scores

try:
	from tqdm import tqdm
except ImportError:
	def tqdm(gen, *args, **kwargs):
		return gen

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

	if kwargs.get('only_one_training'):
		return None

	# Beginning of the second training from the training images

	train_indexes = np.unique(train_labels[:,0])
	predictions = predict(clf, images, box_size, vectorize, only=train_indexes)

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
	limit_score = kwargs.get('limit_score', LIMIT_SCORE)
	slide_step = kwargs.get('slide_step', (20, 20))
	downscale_step = kwargs.get('downscale_step', 0)
	only = kwargs.get('only', None)

	predictions = []
	for index, image in enumerate(tqdm(images, desc='Predicting')):
		if only is not None and index not in only:
			continue

		coordinates, windows = sliding_windows(image, box_size, slide_step, downscale_step)

		# Get the set and predict scores per class
		X = vectorize(windows, *kwargs.get('vectorize_args', []))
		scores = get_scores(clf, X)

		prediction = filter_window_results(index+1, coordinates, scores, limit_score)
		predictions.extend(prediction)

		# TODO validation
			
	return np.array(predictions)

def predict_and_validate(clf, images, box_size, test_labels, vectorize, **kwargs):
	"""
	@brief      Compute the accuracy of the trained classifier only on labelled boxes
	
	@param      clf           The trained classifier
	@param      images        The images
	@param      box_size      The box size
	@param      test_labels   The test labels
	@param      vectorize     The function used to vectorize the windows for the classifier
	
	@return     The covering boxes with their scores: [[ img_id, x, y, h, l, s ]]
	"""
	limit_score = kwargs.get('limit_score', LIMIT_SCORE)
	boxes = extract_boxes(images, test_labels, box_size)

	X = vectorize(boxes, *kwargs.get('vectorize_args', []))
	scores = get_scores(clf, X)
	results = get_results_from_scores(scores, test_labels, limit_score)

	return scores, results
