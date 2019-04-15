import numpy as np
from .utils import area_rate

def getValidRate(predictions, labels):
    """
        @brief Compute the error rate of predictions based on labels
    
        @param  predictions     Each prediction must be like this [image_id,x,y,h,l,score]
        @param  labels          A label is like this [image_id,x,y,h,l,class]
    
    """

    for prediction in predictions:

    #     print(labels)
        
        recognized_faces = 0

        image_id = prediction[0]
        print(image_id)

        label_indices = np.where(labels[:,0] == image_id)

        for label_indice in label_indices:
            

            if area_rate(labels[label_indice][0][1:5],prediction[1:5]) > 1/2 :
                recognized_faces += 1
                break
    print(recognized_faces)
    print(len(predictions))
    print("Taux de réussite : " + str(recognized_faces/len(predictions)) )