from modules.negative_set import generate_negative_set
from modules import data
import numpy as np

def main():
	print("Loading data...")
	train_images = data.load_images()
	labels = data.load_labels()

	print("Generating negative set...")
	neg_set = generate_negative_set(train_images, labels, set_size=300)

	all_labels = np.concatenate([labels, neg_set])
	faces = data.extract_faces(train_images, labels)

	print("Test now !")
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	main()