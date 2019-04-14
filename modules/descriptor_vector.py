import numpy as np
from skimage.feature import hog
from modules import data

SHAPE_H = 40
SHAPE_L = 40

def extract_faces(images, labels):
	
    # Extract faces
    faces = []
    current_idx = None
    current_img = None

    for idx, x, y, h, l, _ in labels:
		# Get image if needed
        if idx != current_idx:
            current_idx = idx
        current_img = images[int(idx)-1]

		# # Extract face, resize and apply Hog        
        img = current_img[int(x):int(x+h),int(y):int(y+l)]
        img_resize = data.compress_image(img, size=(SHAPE_H,SHAPE_L))
        faces.append(hog(img_resize))

    return np.array(faces)