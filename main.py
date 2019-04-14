from modules.negative_set import generate_negative_set, get_box_parameters
from modules import data, models
import numpy as np
from skimage.transform import resize


def main():
	LIMIT = 100
	TRAIN_RATE = 0.75
	clf = models.create_model()

	print("Loading data...")
	images = data.load_images(limit=LIMIT)
	labels = data.load_labels(limit=LIMIT)
	box_size = get_box_parameters(labels)[1:3]

	print("Generating negative set...")
	negatives = generate_negative_set(images, labels, set_size=LIMIT)
	all_labels = np.concatenate([labels, negatives])

	print("Creating train & validation sets...")
	train_labels, valid_labels = data.train_valid_sets(len(images), all_labels)

	# Fake vectorize function
	def fake_vect(images, size):
		return np.array([ resize(img, size, mode='constant', anti_aliasing=True).flatten()[:500] for img in images ])

	print("Training...")
	models.train(clf, images, train_labels, vectorize=fake_vect, vectorize_args=[box_size])

	print("Get validation accuracy...")
	accuracy = models.try_accuracy(clf, images, valid_labels, vectorize=fake_vect, vectorize_args=[box_size])
	print("Accuracy:", accuracy)

	print("Predicting...")
	predictions = models.predict(clf, images, box_size, vectorize=fake_vect, vectorize_args=[box_size])


	print("Test now !")
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	main()