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

	print("Creating train & validation sets...")
	# train_images, train_labels, valid_images, valid_labels = data.train_valid_sets(images, labels)
	# TODO : Pb of index with sets and labels
	train_images = valid_images = images
	train_labels = valid_labels = labels

	print("Generating negative set...")
	neg_set = generate_negative_set(images, labels, set_size=LIMIT)

	# Fake vectorize function
	def fake_vect(images, size):
		return np.array([ resize(img, size, mode='constant', anti_aliasing=True).flatten()[:5000] for img in images ])

	print("Training...")
	models.train(clf, train_images, train_labels, vectorize=fake_vect, negatives=None, vectorize_args=[box_size])

	print("Predicting...")
	predictions = models.predict(clf, valid_images, box_size, vectorize=fake_vect, vectorize_args=[box_size])


	print("Test now !")
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	main()