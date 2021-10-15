# inference.py
# author:Ren
# Date: 20211001

import os
import cv2
import json
import base64
import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from scipy.ndimage import zoom
from sklearn.metrics import classification_report 
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# preprocessing
def crop_center(img):
    y, x = img.shape
    crop_size = np.min([y,x])
    startx = x // 2 - (crop_size // 2)
    starty = y // 2 - (crop_size // 2)
    return img[starty:starty + crop_size, startx:startx + crop_size]

def de_crop_center(img, cam):
    y, x, _ = img.shape
    crop_size = np.min([y,x])
    startx = x // 2 - (crop_size // 2)
    starty = y // 2 - (crop_size // 2)
    padding = [128, 0, 0] * np.ones(img.shape)
    padding[starty:starty + crop_size, startx:startx + crop_size] = cam
    return padding

def torchxrayvision_normalize(img, maxval=255, reshape=False):
    """Scales images to be roughly [-1024 1024]."""
    
    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(img.max(), maxval))
    
    img = (2 * (img.astype(np.float32) / maxval) - 1.) * 1024

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # add color channel
        img = img[None, :, :] 
    
    return img

def de_torchxrayvision_normalize(img):
    new = ((img/1024+1)*255)/2
    return new

def img_preprocess(img_str, type_):
    #img = cv2.imread(file_path, 0)
    de_img = base64_decode(img_str)
    # rgb2Gray
    img = cv2.cvtColor(de_img, cv2.COLOR_RGB2GRAY)
    if type_ == "tube":
        # center crop
        img = crop_center(img)
        # clahe 
        clahe = cv2.createCLAHE()
        img = clahe.apply(img)
    if type_ == "pneumo":
        # torchxrayvision_normalize
        img = torchxrayvision_normalize(img)
    # extend from gray scale to 3 channels
    img = np.array([img, img, img]).transpose(1,2,0)
    # resize to (512, 512, 3)
    new_shape = (512, 512)
    zoom_rate = (new_shape[0]/img.shape[0], new_shape[1]/img.shape[1], 1)
    img = zoom(img, zoom = zoom_rate, order = 3)
    if type_ == "tube":
        # normalize to 0~1
        img = img/255.
    return img

def grad_cam(input_model, image, img_shape, layer_name):
    grad_model = Model(input_model.inputs, [input_model.get_layer(layer_name).output, input_model.output])
    grad_model.layers[-1].activation = tf.keras.activations.linear
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([image]))
        loss = predictions[:]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    grads_val = grads.numpy()

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.dot(output, weights)

    cam = cv2.resize(cam, img_shape)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    
    #return np.uint8(cam), heatmap
    return cam.astype(int), heatmap

def de_shape_cam(new_shape, cam):
    zoom_rate = (new_shape[0]/cam.shape[0], new_shape[1]/cam.shape[1], 1)
    cam = zoom(cam, zoom = zoom_rate, order = 3)
    return cam

def base64_encode(img):
    _, im_arr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64

def base64_decode(img_str):
    im_bytes = base64.b64decode(img_str)
    im_arr = np.frombuffer(im_bytes, dtype=np.uint8)  # im_arr is one-dim Numpy array
    img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
    return img

def chest_inference(img_str, tube_model_pred, pneumo_model_pred):
    # parameters
    layer_name = "conv5_block32_2_conv"
    result = {}
    # decode im_64 to rgb
    img = base64_decode(img_str)
    # tube inference
    tube_img = img_preprocess(img_str, "tube")
    tube_pred = tube_model_pred.predict(np.expand_dims(tube_img, axis=0))
    result['tube'] = {"pred": float(tube_pred), "mask": ""}
    print("finish tube predict")
    # pneumo inferennce
    pneumo_img = img_preprocess(img_str, "pneumo")
    pneumo_pred = pneumo_model_pred.predict(np.expand_dims(pneumo_img, axis=0))
    result['pneumo'] = {"pred": float(pneumo_pred), "mask": ""}
    print("finish pneumo predict")
    if result['tube']['pred']>=0.5:
        cam, _ = grad_cam(tube_model_pred, tube_img, tube_img.shape[:2], layer_name)
        new_shape = (min(img.shape[:2]), min(img.shape[:2]))
        cam_reshape = de_shape_cam(new_shape, cam)
        cam_padding = de_crop_center(img, cam_reshape)
        overcam_reshape = cv2.addWeighted(img.astype(int), 0.8, cam_padding.astype(int), 0.3, 0)[:,:,::-1]
        result['tube']['mask'] = base64_encode(overcam_reshape).decode('utf-8')
        
    if result['pneumo']['pred']>=0.5:
        cam, _ = grad_cam(pneumo_model_pred, pneumo_img, pneumo_img.shape[:2], layer_name)
        pneumo_img = de_torchxrayvision_normalize(pneumo_img[:,:,0])
        pneumo_img = np.array([pneumo_img, pneumo_img, pneumo_img]).transpose(1,2,0)
        cam_reshape = de_shape_cam(img.shape[:2], cam)
        overcam_reshape = cv2.addWeighted(img.astype(int), 0.8, cam_reshape, 0.3, 0)[:,:,::-1]
        result['pneumo']['mask'] = base64_encode(overcam_reshape).decode('utf-8')

    return json.dumps(result)
