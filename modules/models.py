from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

import numpy as np
from .negative_set import get_box_parameters
from .window import sliding_windows
from .data import extract_boxes

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
		'n_estimators': 10,
	},
}

DECISION_METHODS = {
	'SVC': 										'decision_function',
	'LinearSVC': 							'decision_function',
	'AdaBoostClassifier' : 		'decision_function',
	'DecisionTreeClassifier': 'predict_proba',
	'RandomForestClassifier': 'predict_proba',
}

def create_model(class_name=BEST_MODEL, *args, **params):
	"""Easy constructor for models with default optimized params"""
	if class_name not in MODELS:
		raise NotImplementedError(
						f"Classifier {class_name} is not implemented in this function"
						f"\nYou can use: {', '.join(MODELS.keys())}")

	if not params:
		params = DEFAULT_PARAMS[class_name]

	return MODELS[class_name](*args, **params)

def get_decision(clf, *args, **kwargs):
	"""Get the decision function of a classifier"""
	name = DECISION_METHODS[clf.__class__.__name__]
	method = getattr(clf, name)
	return method(*args, **kwargs)



def train(clf, images, labels, vectorize=lambda boxes: boxes, negatives=None):
	"""
	@brief      Train a classifier with the boxes labelled on the images
	
	@param      clf        The classifier instance
	@param      images     The images
	@param      labels     The labels
	@param      vectorize  The function used to vectorize the extracted boxes of images
	@param      negatives  The negatives labels
	"""

	# Extract boxes of the images from the labels
	all_labels = np.concatenate([labels, negatives]) if negatives is not None else labels
	boxes = extract_boxes(images, all_labels)

	# Get the training set
	X = vectorize(boxes)
	y = all_labels[:,5]

	# Finally, train
	clf.fit(X, y)


	



def predict(clf, images):
	slide_step = (20, 20)
	box_size = get_box_parameters(labels)[1:3]

	results = []
	for image in images:
		for img in down(image):
			windows = slide_windows(image, box_size, slide_step, downscale_step=0)
			# TODO
			predictions = clf.predict(windows[:,2])
			results.append(filter_window_results(predictions))

	return all_boxes

