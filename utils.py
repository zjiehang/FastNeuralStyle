# coding : utf-8
import scipy.misc
import random
import numpy as np

def resize_to(img, resize=512):
    '''Resize short side to target size and preserve aspect ratio'''
    height, width = img.shape[0], img.shape[1]
    if height < width:
        ratio = height / resize
        long_side = round(width / ratio)
        resize_shape = (resize, long_side, 3)
    else:
        ratio = width / resize
        long_side = round(height / ratio)
        resize_shape = (long_side, resize, 3)

    return scipy.misc.imresize(img, resize_shape)


def get_img_random_crop(img, resize=512, crop=256,is_norm=True):
    '''Getresize image and random crop'''
    img = resize_to(img, resize=resize)

    if is_norm:
        offset_h = random.randint(0, (img.shape[0]-crop)-1)
        offset_w = random.randint(0, (img.shape[1]-crop)-1)

        img = img[offset_h:offset_h+crop, offset_w:offset_w+crop, :]
    else:
        h_offset = int(np.floor((img.shape[0] - crop) / 2.))
        w_offset = int(np.floor((img.shape[1] - crop) / 2.))
        h_end = h_offset + crop
        w_end = w_offset + crop
        img = img[h_offset: h_end, w_offset: w_end,:]

    return img


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