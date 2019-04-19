from modules import data, models
import numpy as np
from config import (
	CLASSIFIER, MODEL_PARAMS, BOX_SIZE, KW_PARAMS,
	PREDICTION_PATH, TEST_PATH, MODEL_PATH, GRAY
)

print("Loading model...")
clf = data.load_model(path=MODEL_PATH)

print("Loading test images...")
images = data.load_images(path=TEST_PATH, gray=GRAY)

print("Predicting...")
predictions = models.predict(clf, images, BOX_SIZE, filter=True, **KW_PARAMS)

print("Saving predictions...")
data.save_predictions(predictions, path=PREDICTION_PATH)

print(f"\nTest images predicted and saved in {PREDICTION_PATH} !")
