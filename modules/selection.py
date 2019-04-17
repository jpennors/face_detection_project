from .negative_set import get_box_parameters
from . import models

def try_params(images, label_sets, clf_name, global_params, changing_params, **kwargs):
	"""
	@brief      Multiple testing using changing and global parameters on one classifier
	
	@param      images           The images
	@param      label_sets       The label sets (training and validation)
	@param      clf_name         The classifier name
	@param      global_params    The global parameters used for the classifier and the vectorization
	@param      changing_params  The changing parameters used for the classifier and the vectorization
	
	@return     The results of the multiple tests
	"""
	assert len(label_sets) == 2, "Please provide training and validation labels"
	train_labels, valid_labels = label_sets

	windows_sets = kwargs.get('windows_sets')
	global_vectorization_params = global_params.pop('vectorization_params')
	global_box_size = global_params.pop('box_size')
	param_results = []

	# Try each changing parameter
	for param_name in changing_params:
		print(f"## Trying parameter `{param_name}`...")
		for param_value in changing_params[param_name]:
			print(f"### with value `{param_value}`")

			# Build parameters with one changing
			if param_name == 'vectorization_params':
				box_size = global_box_size
				model_params = global_params
				vectorization_params = param_value
			elif param_name == 'box_size':
				box_size = param_value
				model_params = global_params
				vectorization_params = param_value				
			else:
				box_size = global_box_size
				model_params = {
					**global_params,
					param_name: param_value,
				}
				vectorization_params = global_vectorization_params

			# Build, train and validate classifier
			clf = models.create_model(clf_name, model_params)
			models.train(clf, images, box_size, train_labels, **vectorization_params, windows_sets=windows_sets)
			score, result = models.predict_and_validate(clf, images, box_size, valid_labels,
																									**vectorization_params, windows_sets=windows_sets)

			# Add score to array
			param_results.append({
				'name': param_name,
				'value': param_value,
				'score': score,
				'result': result,
			})

	return {
		'classifier': clf_name,
		'global_params': global_params,
		'results': param_results,
	}

def try_classifiers(images, label_sets, global_params, changing_params={}, **kwargs):
	"""
	@brief      Multiple testing using changing and global parameters on multiple classifiers
	
	@param      images           The images
	@param      label_sets       The label sets (training and validation)
	@param      global_params    The global parameters used for each classifier and the vectorization
															 with the classifier name as key
	@param      changing_params  The changing parameters used for the classifier and the vectorization
															 with the classifier name as key
	
	@return     The results of the multiple tests
	"""
	assert len(label_sets) == 2, "Please provide training and validation labels"
	train_labels, valid_labels = label_sets

	# Process each classifier
	results = {}
	for clf_name in global_params:
		print(f"# Trying classifier `{clf_name}`...")
		g_params = global_params[clf_name]
		c_params = changing_params.get(clf_name)

		if c_params:
			results[clf_name] = try_params(images, label_sets, clf_name, g_params, c_params, **kwargs)
		else:
			box_size = g_params['box_size']
			vectorization_params = g_params['vectorization_params']
			model_params = { key: value for key, value in g_params.items()
											 if key not in ('box_size', 'vectorization_params')}

			clf = models.create_model(clf_name, model_params)
			models.train(clf, images, box_size, train_labels, **vectorization_params)
			score, result = models.predict_and_validate(clf, images, box_size, valid_labels, **vectorization_params)
			results[clf_name] = {
				'classifier': clf_name,
				'global_params': global_params,
				'results': [{
					'score': score,
					'result': result,
				}],
			}

	return results
