from modules.negative_set import generate_negative_set, get_box_parameters
from modules import data, models
import numpy as np
from skimage.transform import resize

def main():
	LIMIT = 100
	clf = models.create_model()

	print("Loading data...")
	images = data.load_images(limit=LIMIT)
	labels = data.load_labels(limit=LIMIT)
	box_size = get_box_parameters(labels)[1:3]

	print("Generating negative set...")
	neg_set = generate_negative_set(images, labels, set_size=LIMIT)

	# Fake vectorize function
	def fake_vect(images):
		return np.array([ resize(img, box_size, mode='constant', anti_aliasing=True).flatten()[:5000] for img in images ])

	print("Training...")
	models.train(clf, images, labels, vectorize=fake_vect, negatives=None)



	print("Test now !")
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	main()