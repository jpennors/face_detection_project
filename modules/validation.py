import numpy as np
from .utils import area_rate

def apply_first_validation(predictions, labels):
    """
    @brief Compute the error rate of predictions based on labels

    @param  predictions     Each prediction must be like this [image_id,x,y,h,l,score]
    @param  labels          A label is like this [image_id,x,y,h,l,class]
    """

    recognized_faces = 0
    false_positive = []

    for prediction in predictions:
        
        image_id = prediction[0]

        label_indices = np.where(labels[:,0] == image_id)

        for label_indice in label_indices:
            
            valid = False
            if area_rate(labels[label_indice][0][1:5],prediction[1:5]) > 1/2 :
                recognized_faces += 1
                valid = True
                break

        if not valid:
            # Change format, to become a label (score -> class = -1)
            # prediction[5] = -1
            false_positive.append([prediction[0], prediction[1], prediction[2], prediction[3], prediction[4], -1])
    print(false_positive)
    print(recognized_faces)
    print(len(predictions))
    print("Taux de réussite : " + str(recognized_faces/len(predictions)) )

    return np.array(false_positive)