from modules import descriptor_vector

# Paths
LABEL_PATH = 'data/label.txt'
TRAIN_PATH = 'data/train/'
TEST_PATH  = 'data/test/'
PREDICTION_PATH = 'detection.txt'
MODEL_PATH = 'model.pickle'

PREDICTION_PATH = 'detection1.txt'
MODEL_PATH = 'model1.pickle'

# Classifiers
CLASSIFIER = 'linear_svc'
CLASSIFIER = 'random_forest'
MODEL_PARAMS = {
	'n_estimator': 150,
	# 'C': 0.75,
	# 'max_iter': 10000,
}
VECTORIZATION_PARAMS = {
	'vectorize': descriptor_vector.hog,
	# 'vectorize_args': [6],
} 

# Labels
LIMIT = None
OFFSET = 0
GRAY = True
TEST_LIMIT = 100

NEG_SIZE = 4000
BOX_SIZE = (150,90)
SLIDE_STEP = (60, 50)
DOWNSCALE_STEP = 30
LIMIT_SCORE = 0.5

KW_PARAMS = {
	**VECTORIZATION_PARAMS,
	'limit_score': LIMIT_SCORE,
	'slide_step': SLIDE_STEP,
	'downscale_step': DOWNSCALE_STEP,
}
