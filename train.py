from modules.negative_set import generate_negative_set, get_box_parameters
from modules import data, models
import numpy as np
from config import (
	PREDICTION_PATH, MODEL_PATH, TRAIN_IMAGES_PATH, LABEL_PATH,
	CLASSIFIER, MODEL_PARAMS, BOX_SIZE, KW_PARAMS,
	LIMIT, OFFSET, GRAY, NEG_SIZE,
)

clf = models.create_model(CLASSIFIER, MODEL_PARAMS)

print("Loading data...")
images = data.load_images(path=TRAIN_IMAGES_PATH, limit=LIMIT, offset=OFFSET, gray=GRAY)
labels = data.load_labels(path=LABEL_PATH, limit=LIMIT, offset=OFFSET)

print("Generating negative set...")
negatives = generate_negative_set(images, labels, set_size=NEG_SIZE)
all_labels = np.concatenate([labels, negatives])

print("Training...")
models.train(clf, images, BOX_SIZE, all_labels, **KW_PARAMS)

print("Saving model...")
data.save_model(clf, path=MODEL_PATH)

print(f"\nModel trained and saved in {MODEL_PATH} !")
