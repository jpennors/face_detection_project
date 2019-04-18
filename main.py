from modules.negative_set import generate_negative_set, get_box_parameters
from modules import data, models, descriptor_vector, validation, selection
import numpy as np
from skimage.transform import resize
import pickle


# Params
TRAIN = True
SAVE_MODEL = False
SAVE_NEGATIVES = False
LIMIT = 200
NEG_SIZE = 60
TRAIN_RATE = 0.70
CLASSIFIER = 'random_forest'
MODEL_PARAMS = {
	'n_estimators': 100,
}
VECTORIZATION_PARAMS = {
	'vectorize': descriptor_vector.hog,
	# 'vectorize_args': [box_size],
}

def main():
	if TRAIN:
		clf = models.create_model(CLASSIFIER, MODEL_PARAMS)

		print("Loading data...")
		images = data.load_images(limit=LIMIT)
		labels = data.load_labels(limit=LIMIT)
		
		print("Params:")
		box_size = get_box_parameters(labels)[1:3]
		box_size = box_size[0] - 10, box_size[1] - 10
		print(" ", clf.__class__.__name__)
		print(" ", MODEL_PARAMS)
		print(" ", LIMIT, "images,", NEG_SIZE, "negatives")
		print("  box_size:", box_size)
		print(" ", VECTORIZATION_PARAMS)


		print("Generating negative set...")
		negatives = generate_negative_set(images, labels, set_size=NEG_SIZE, save=SAVE_NEGATIVES)
		all_labels = np.concatenate([labels, negatives])

		print("Creating train & validation sets with negatives...")
		train_labels, valid_labels = data.train_valid_sets(len(images), all_labels, TRAIN_RATE)

		print("Training...")
		models.train(clf, images, box_size, train_labels, **VECTORIZATION_PARAMS)

		import pdb; pdb.set_trace()

		print("\nPredicting with windows...")
		valid_indexes = np.unique(valid_labels[:,0]) - 1
		predictions = models.predict(clf, images, box_size, **VECTORIZATION_PARAMS, only=valid_indexes)

		if SAVE_MODEL:
			import pdb; pdb.set_trace()
			print('Saving alllll')
			to_save = [clf, images, box_size, train_labels, valid_labels, predictions]
			model_file = open('./temp.pickle', 'wb')
			pickle.dump(to_save, model_file)

	else:
		print("Loading all..")
		model_file = open('./temp.pickle', 'rb')
		clf, images, box_size, train_labels, valid_labels, predictions = pickle.load(model_file)


	# print("Get validation classification accuracy...")
	# accuracy = models.accuracy(clf, images, box_size, valid_labels, **VECTORIZATION_PARAMS)
	# print("  Accuracy:", accuracy)

	# print("Predicting and validate on test examples...")
	# scores, results = models.predict_and_validate(clf, images, box_size, valid_labels, **VECTORIZATION_PARAMS)

	print("\nPredicting with windows...")
	valid_indexes = np.unique(valid_labels[:,0]) - 1
	predictions = models.predict(clf, images, box_size, **VECTORIZATION_PARAMS, only=valid_indexes)

	import pdb; pdb.set_trace()

	print("\nPredicting with windows and validate...")
	results = validation.rate_predictions(predictions, valid_labels)

	print("Test now !")
	import pdb; pdb.set_trace()


if __name__ == '__main__':
	main()