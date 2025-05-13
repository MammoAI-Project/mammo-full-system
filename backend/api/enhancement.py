import numpy as np
import tensorflow as tf
import cv2

def apply_clahe_enhancement(images):
    enhanced_images = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for img in images:
        lab = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        lab_planes = list(cv2.split(lab))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) / 255.0
        enhanced_images.append(enhanced)

    return np.array(enhanced_images)