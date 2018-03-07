# coding : utf-8
import scipy.misc
import random
import numpy as np
import tensorflow as tf
import scipy.misc


def resize_to(img, resize_shape=512):
    #Resize short side to target size and preserve aspect ratio
    if img.size == 1:
        height, width = img.item(0).size
        if height < width:
            ratio = height / resize_shape
            long_side = round(width / ratio)
            resize_shape = (resize_shape, long_side)
        else:
            ratio = width / resize_shape
            long_side = round(height / ratio)
            resize_shape = (long_side, resize_shape)

        img_object = img.item(0)
        img_after_resize = img_object.resize(resize_shape)
        return np.array(img_after_resize)

    else:
        height, width = img.shape[0], img.shape[1]
        if height < width:
            ratio = height / resize_shape
            long_side = round(width / ratio)
            resize_shape = (resize_shape, long_side, 3)
        else:
            ratio = width / resize_shape
            long_side = round(height / ratio)
            resize_shape = (long_side, resize_shape, 3)
        return scipy.misc.imresize(img, resize_shape)
'''
def get_img_random_crop(img, resize=512, crop=256,is_random=True):
    #Getresize image and random crop
    img = resize_to(img, resize=resize)

    if is_random:
        offset_h = random.randint(0, (img.shape[0]-crop-1))
        offset_w = random.randint(0, (img.shape[1]-crop-1))

        img = img[offset_h:offset_h+crop, offset_w:offset_w+crop,:]
    else:
        h_offset = int(np.floor((img.shape[0] - crop) / 2.))
        w_offset = int(np.floor((img.shape[1] - crop) / 2.))
        h_end = h_offset + crop
        w_end = w_offset + crop
        img = img[h_offset: h_end, w_offset: w_end,:]

    return img
'''

def get_vgg19_decoder_layers_detail(content_encoder_layer):
    vgg19_layers = [2,2,4,4,4]

    content_encoder_layer = content_encoder_layer.replace('conv','')
    layers_detail_list = content_encoder_layer.split('_')
    layers,last_layer_number = int(layers_detail_list[0]),int(layers_detail_list[1])

    layers_detail = []
    for i in range(layers-1):
        layers_detail.append(vgg19_layers[i])
    layers_detail.append(last_layer_number)
    return layers_detail


def mean_squared(x,y):
    return tf.reduce_mean(tf.square( x - y ))

def gram_matrix(feature_maps):
    """Computes the Gram matrix for a set of feature maps.
       Borrowed from https://github.com/tensorflow/magenta/blob/9eb2e71074c09f55dba10cc493d26aef3168cdcb/magenta/models/image_stylization/learning.py
    """
    batch_size, height, width, channels = tf.unstack(tf.shape(feature_maps))
    denominator = tf.to_float(height * width)
    feature_maps = tf.reshape(
      feature_maps, tf.stack([batch_size, height * width, channels]))
    matrix = tf.matmul(feature_maps, feature_maps, adjoint_a=True)
    return matrix / denominator


def learning_rate_decay(learning_rate, global_step, decay_rate, name=None):
    '''Adapted from https://github.com/torch/optim/blob/master/adam.lua'''
    if global_step is None:
        raise ValueError("global_step is required for exponential_decay.")
    with tf.name_scope(name, "ExponentialDecay", [learning_rate, global_step, decay_rate]) as name:
        learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = tf.cast(global_step, dtype)
        decay_rate = tf.cast(decay_rate, dtype)

        # local clr = lr / (1 + state.t*lrd)
        return learning_rate / (1 + global_step*decay_rate)

def normalize_color(img):
    return tf.div(img,tf.constant(255.0,dtype=tf.float32))