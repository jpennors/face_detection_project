# Intersection area functions
# A box is like this [ x, y, h, l ]

def area(box):
	return box[2] * box[3]

def area_between(box_1, box_2):
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
	a_in = area_between(box_1, box_2)
	if a_in == 0:
		return 0
	return a_in / (area(box_1) + area(box_2) - a_in)
