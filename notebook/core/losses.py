# Author: Ankit Kariryaa, University of Bremen
# Modified by Beihui Hu

import tensorflow.keras.backend as K


def dice_coef(y_true, y_pred):
    """compute dice coef"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

def dice_loss(y_true, y_pred):
    """compute dice loss"""
#     y_t = y_true[...,0]
#     y_t = y_t[...,np.newaxis]
    return 1 - dice_coef(y_true, y_pred)


def true_positives(y_true, y_pred):
    """compute true positive"""
#     y_t = y_true[...,0]
#     y_t = y_t[...,np.newaxis]
    y_t = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.round(y_t * y_pred)

def false_positives(y_true, y_pred):
    """compute false positive"""
#     y_t = y_true[...,0]
#     y_t = y_t[...,np.newaxis]
    y_t = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.round((1 - y_t) * y_pred)

def true_negatives(y_true, y_pred):
    """compute true negative"""
    y_t = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.round((1 - y_t) * (1 - y_pred))

def false_negatives(y_true, y_pred):
    """compute false negative"""
    y_t = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.round((y_t) * (1 - y_pred))

def accuracy(y_true, y_pred):#Calculates how often predictions equal labels.
    """compute accuracy"""
    y_t = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return K.equal(K.round(y_t), K.round(y_pred))#Accuracy =  (TP + TN) / (TP + TN + FP+ FN) 

def IoU(y_t, y_pred):#the Intersection-Over-Union metric.
    # IoU = TP / (TP + FP + FN)
    tp = true_positives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp)/(K.sum(tp)+K.sum(fp)+K.sum(fn)+K.epsilon())
    
def recall(y_t, y_pred):#recall = TP / (TP + FN)
    """compute sensitivity (recall)"""
    tp = true_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn)+ K.epsilon())

def precision(y_t, y_pred):
    """compute precision"""
    tp = true_positives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fp)+ K.epsilon())

def F1_score(y_t, y_pred):
    re = recall(y_t, y_pred)
    pr = precision(y_t, y_pred)
    return 2*pr*re/(re+pr+K.epsilon())