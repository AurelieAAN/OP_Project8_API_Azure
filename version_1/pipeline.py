import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import imgaug as ia
from imgaug import augmenters as iaa
from matplotlib import pyplot as plt
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence
import imgaug as ia
from imgaug import augmenters as iaa

def iou(y_true,y_pred):
  def f(y_true,y_pred):
    intersection = (y_true*y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    x = (intersection + 1e-15) / (union + 1e-15)
    x = x.astype(np.float32)
    return x
  return tf.numpy_function(f,[y_true,y_pred],tf.float32)

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)

reconstructed_model = tf.keras.models.load_model("best_model.h5",custom_objects={'dice_coef_loss': dice_coef_loss, 'iou':iou, 'tversky':tversky})

def form_colormap(prediction,mapping):
    h,w = prediction.shape
    color_label = np.zeros((h,w,3),dtype=np.uint8)    
    color_label = mapping[prediction]
    color_label = color_label.astype(np.uint8)
    return color_label


class_map_dict = {
        "void": [155, 155, 155],
        "flat": [60, 16, 152],
        "construction": [132, 41, 246],
        "object": [110, 193, 228],
        "nature": [254, 221, 58],
        "sky": [226, 169, 41],
        "human": [254, 200, 58],
        "vehicle": [220, 100, 41]
    }
class_map = []
for key, values in class_map_dict.items():
    class_map.append(values)


def make_prediction(model,img_path,shape):
    img= tf.keras.utils.img_to_array(tf.keras.utils.load_img(img_path,  target_size= shape))
    img = np.expand_dims(img,axis=0)
    labels = model.predict(img)
    labels = np.argmax(labels[0],axis=2)
    return labels

def predict(image):
    pred_label = make_prediction(reconstructed_model, image, (224,224,3))
    pred_colored = form_colormap(pred_label,np.array(class_map))
    return pred_colored