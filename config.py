from modules import descriptor_vector

VERSION = '2'

# Paths
LABEL_PATH = 'data/label.txt'
TRAIN_PATH = 'data/train/'
TEST_PATH  = 'data/test/'
PREDICTION_PATH = 'detection.txt'
MODEL_PATH = 'model.pickle'

PREDICTION_PATH = f'detection{VERSION}.txt'
MODEL_PATH = f'model{VERSION}.pickle'

# Classifiers
# CLASSIFIER = 'linear_svc'
CLASSIFIER = 'random_forest'
MODEL_PARAMS = {
	'n_estimators': 200,
	# 'C': 0.75,
	# 'max_iter': 10000,
}
VECTORIZATION_PARAMS = {
	'vectorize': descriptor_vector.daisy,
	# 'vectorize_args': [6],
} 

# Labels
LIMIT = None
OFFSET = 0
GRAY = True
TEST_LIMIT = None

NEG_SIZE = 4000
BOX_SIZE = (100,65)
SLIDE_STEP = (30, 30)
DOWNSCALE_STEP = 30
LIMIT_SCORE = 0.5

KW_PARAMS = {
	**VECTORIZATION_PARAMS,
	'limit_score': LIMIT_SCORE,
	'slide_step': SLIDE_STEP,
	'downscale_step': DOWNSCALE_STEP,
}
