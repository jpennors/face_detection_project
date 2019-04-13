from modules.negative_set import generate_negative_set
from modules import data

def main():
	print("Loading data...")
	labels = data.load_labels()
	train_images = data.load_images(limit=400)

	print("Generating negative set...")
	neg_set = generate_negative_set(train_images, labels, set_size=300)

	print("Test now !")
	import pdb; pdb.set_trace()

if __name__ == '__main__':
	main()