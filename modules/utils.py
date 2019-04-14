import numpy as np

def get_shape_stats(shapes, display_info=False):
	"""Compute statistics about image or box shapes [[h, l]]"""
	h_col = shapes[:,0]
	l_col = shapes[:,1]
	r_col = h_col / l_col
	h_mean, h_std = np.mean(h_col), np.std(h_col)
	l_mean, l_std = np.mean(l_col), np.std(l_col)
	r_mean, r_std = np.mean(r_col), np.std(r_col)

	if display_info:
		print(f"Ration moyen h/l: {r_mean:6.2f} +/- {r_std:5.2f}")
		print(f"Hauteur moyenne : {h_mean:6.2f} +/- {h_std:5.2f}")
		print(f"Largeur moyenne : {l_mean:6.2f} +/- {l_std:5.2f}")

	return r_mean, int(h_mean), int(l_mean), r_std, h_std, l_std


# Intersection area functions
# A box is like this [ x, y, h, l ]
def area(box):
	"""Compute are of a box like this [x, y, h, l]"""
	return box[2] * box[3]

def cover_aera(box_1, box_2):
	"""Compute overlap between two boxes like this [x, y, h, l]"""
	# s: start, e: end
	xs_1, ys_1, h1, l1 = box_1
	xs_2, ys_2, h2, l2 = box_2
	xe_1, ye_1 = xs_1 + h1, ys_1 + h1
	xe_2, ye_2 = xs_2 + h2, ys_2 + h2

	# No aera between
	if xe_1 < xs_2 or xs_1 > xe_2 or ye_1 < ys_2 or ys_1 > ye_2:
		return 0
	xs = max(xs_1, xs_2)
	xe = min(xe_1, xe_2)
	ys = max(ys_1, ys_2)
	ye = min(ye_1, ye_2)
	return (xe - xs) * (ye - ys)

def area_rate(box_1, box_2):
	"""Compute are rate between two boxes like this [x,y,h, l]"""
	a_in = cover_aera(box_1, box_2)
	if a_in == 0:
		return 0
	return a_in / (area(box_1) + area(box_2) - a_in)
