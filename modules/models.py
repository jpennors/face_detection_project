from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from .negative_set import generate_negative_set


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
	'random_forest': {},
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


def train(clf, images, labels, negatives=None):
	
	for image in images:
		pass



def predict(clf, images):
	all_boxes = []
	for image in images:
		for img in down(image):
			boxes = slide_window(clf, img, step=DEFAULT_STEP, threshold=0.2)
			boxes = filter_cover_boxes(boxes)
			all_boxes.append(boxes)

	return all_boxes

