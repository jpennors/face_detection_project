from modules.negative_set import generate_negative_set
from modules import data, models
import numpy as np

def main():
	LIMIT = 10

	print("Loading data...")
	train_images = data.load_images(limit=LIMIT)
	labels = data.load_labels(limit=LIMIT)

	print("Generating negative set...")
	neg_set = generate_negative_set(train_images, labels, set_size=300)

	print("Extracting examples...")
	all_labels = np.concatenate([labels, neg_set])
	examples = data.extract_faces(train_images, labels)

	clf = models.create_model()
	import pdb; pdb.set_trace()

	print("Training...")
	models.train(clf, train_images, labels, negatives=neg_set)


	print("Test now !")
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	main()