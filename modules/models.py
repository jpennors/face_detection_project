from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import numpy as np
from .negative_set import get_box_parameters
from .window import extract_boxes, sliding_windows
from .window import filter_window_results

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

LIMIT_SCORE = 0.5

def create_model(class_name=BEST_MODEL, params=None):
	"""Easy constructor for models with default optimized params"""
	if class_name not in MODELS:
		raise NotImplementedError(
						f"Classifier {class_name} is not implemented in this function"
						f"\nYou can use: {', '.join(MODELS.keys())}")

	if not params:
		params = DEFAULT_PARAMS[class_name]

	return MODELS[class_name](**params)

def get_decision(clf, *args, **kwargs):
	"""Get the decision function of a classifier"""
	name = DECISION_METHODS[clf.__class__.__name__]
	method = getattr(clf, name)
	return method(*args, **kwargs)



def train(clf, images, box_size, labels, vectorize, negatives=None, **kwargs):
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
	boxes = extract_boxes(images, labels, box_size)

	# Get the training set
	X = vectorize(boxes, *kwargs.get('vectorize_args', []))
	y = labels[:,5]

	# Finally, train
	clf.fit(X, y)

def accuracy(clf, images, box_size, labels, vectorize, negatives=None, **kwargs):
	boxes = extract_boxes(images, labels, box_size)

	# Get the training set
	X = vectorize(boxes, *kwargs.get('vectorize_args', []))
	y = labels[:,5]

	return clf.score(X, y)

def predict(clf, images, box_size, vectorize, **kwargs):
	# Get params
	slide_step = kwargs.get('slide_step', (20, 20))
	downscale_step = kwargs.get('downscale_step', 0)

	results = []
	for index, image in enumerate(images):

		coordinates, windows = sliding_windows(image, box_size, slide_step, downscale_step)

		# Get the set and predict
		X = vectorize(windows, *kwargs.get('vectorize_args', []))
		# import pdb; pdb.set_trace()
		y = get_decision(clf, X)

		# import pdb; pdb.set_trace()
		predictions = filter_window_results(coordinates, y, LIMIT_SCORE, index+1)
		for prediction in predictions:
			results.append(prediction)
			
	return np.array(results)

