import numpy as np
from skimage import io
from skimage import util
from skimage import color
from skimage import feature
import os
from random import randint

def load_label():

    label = np.loadtxt('project_train/label.txt')
    return label

def get_box_parameter(label):
    ratio = 0
    hauteur_moy = 0
    largeur_moy = 0

    # Itération pour faire le ratio hauteur et largeur (h/l)
    for i in range(len(label)):
        ratio += (label[i][3] / label[i][4])/len(label)
        hauteur_moy += label[i][3]/len(label)
        largeur_moy += label[i][4]/len(label)

    print("Le ration entre la hauteur et la largeur est de ", ratio)
    print("La hauteur moyenne est de ", hauteur_moy)
    print("La largeur moyenne est de ", largeur_moy)

    return ratio, int(hauteur_moy), int(largeur_moy)

def load_images():

    path_train = 'project_train/train/'

    img = []

    for element in os.listdir(path_train):
        im = io.imread(path_train + element)
        img.append(im)

    return img


def generate_box(img, hauteur_box, largeur_box):
    
    # Génère hauteur, largeur et position (x,y)
    #h = randint(50,150)
    h = hauteur_box
    # l = int(h/ratio)
    l = largeur_box
    x = randint(0, img.shape[1] - l)
    y = randint(0, img.shape[0] - h)

    return (h,l,x,y)

def generate_negative_img(img, num, hauteur_box, largeur_box):
    
    h2, l2, x2, y2 = generate_box(img, hauteur_box, largeur_box)

    return check_img_content(img, num, h2, l2, x2, y2)


def check_img_content(img, num, h2, l2, x2, y2):

    label = load_label()

    # Récupère les visages trouvés sur l'image
    visages = np.where(label[:,0] == num)[0]

    validation = True

    for i in visages:
        
        h1 = label[i][3]
        l1 = label[i][4]
        x1 = label[i][1]
        y1 = label[i][2]
        
        if(((x1 < x2 + l2 and x1 + l1 > x2) or (x2 < x1 + l1 and x2 + l2 > x1)) and ((y1 < y2 + h2 and y1 + h1 > y2) or (y2 < y1 + h1 and y2 + h2 > y1))):
            pass
        else: 
            xe_inter = max(x1, x2)
            ye_inter = max(y1,y2)

            ys_inter = min(y1+h1, y2+h2)
            xs_inter = min(x1+l1, x2+l2)

            aire_inter = (xs_inter-xe_inter) * (ys_inter-ye_inter)
            aire_union = ((h1*l1) + (h2*l2)) - aire_inter

            if aire_inter / aire_union > 1/2 :
                validation = False                  
    
    if validation:
        return np.array((num, x2, y2, l2, h2))
    else:
        return None


def generate_negative_set():
    

    img = load_images()

    label = load_label()

    ratio_box, hauteur_box, largeur_box = get_box_parameter(label)


    set_img = np.zeros((300,5))
    index = 0

    while index < 300 :
        
        num = randint(1, len(img))
    
#     for index, element in enumerate(os.listdir(path_train)):
#         im = util.img_as_float(io.imread(str(path_train + element)))
        
        res = generate_negative_img(img[num], num, hauteur_box, largeur_box)
        if not (res is None):
            set_img[index] = res
            index += 1

    return set_img
        
#set_img = generate_negative_set(img, hauteur_box, largeur_box)