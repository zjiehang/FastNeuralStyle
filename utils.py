# coding : utf-8
import scipy.ndimage as spi
import scipy.sparse as sps
import numpy as np
import tensorflow as tf
import scipy.misc
import scipy.io

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


def get_channel_number_from_vgg19_layer(layer):
    vgg = {'conv1':64,
           'conv2':128,
           'conv3':256,
           'conv4':512,
           'conv5':512}
    for key in vgg:
        if layer.startswith(key):
            return vgg[key]


def mean_squared(x,y):
    return tf.reduce_mean(tf.squared_difference(x,y))


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


def get_sqarse_tensor_value_by_mat(matpath,imgsize):
    sqarse_tensor_value_list = list()
    for matpatheach in matpath:
        read_in_numpy_array = scipy.io.loadmat(matpatheach)['CSR']
        indices = np.mat([read_in_numpy_array[:,0].astype(np.int32),read_in_numpy_array[:,1].astype(np.int32)]).transpose()
        sqarse_tensor_value = tf.SparseTensorValue(indices,read_in_numpy_array[:,2].astype(np.float32),[imgsize*imgsize,imgsize*imgsize])
        sqarse_tensor_value_list.append(sqarse_tensor_value)
    return sqarse_tensor_value_list

def getlaplacian1(i_arr, consts, epsilon=1e-5, win_rad=1):
    neb_size = (win_rad * 2 + 1) ** 2
    h, w, c = i_arr.shape
    img_size = w * h
    consts = spi.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_rad * 2 + 1, win_rad * 2 + 1)))

    indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
    tlen = int((-consts[win_rad:-win_rad, win_rad:-win_rad] + 1).sum() * (neb_size ** 2))
    row_inds = np.zeros(tlen)
    col_inds = np.zeros(tlen)
    vals = np.zeros(tlen)
    l = 0
    for j in range(win_rad, w - win_rad):
        for i in range(win_rad, h - win_rad):
            if consts[i, j]:
                continue
            win_inds = indsM[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1]
            win_inds = win_inds.ravel(order='F')
            win_i = i_arr[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1, :]
            win_i = win_i.reshape((neb_size, c), order='F')
            win_mu = np.mean(win_i, axis=0).reshape(c, 1)
            win_var = np.linalg.inv(
                np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu, win_mu.T) + epsilon / neb_size * np.identity(
                    c))

            win_i2 = win_i - np.repeat(win_mu.transpose(), neb_size, 0)
            tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size

            ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
            row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
            col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
            vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
            l += neb_size ** 2

    vals = vals.ravel(order='F')[0: l]
    row_inds = row_inds.ravel(order='F')[0: l]
    col_inds = col_inds.ravel(order='F')[0: l]
    a_sparse = sps.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))

    sum_a = a_sparse.sum(axis=1).T.tolist()[0]
    a_sparse = sps.diags([sum_a], [0], shape=(img_size, img_size)) - a_sparse

    return a_sparse

def getLaplacian(imgs):
    if isinstance(imgs, (list,)):
        result = []
        for img in imgs:
            h, w, _ = img.shape
            coo = getlaplacian1(img, np.zeros(shape=(h, w)), 1e-5, 1).tocoo()
            result.append(coo.data)
        return result
    else:
        h, w, _ = imgs.shape
        coo = getlaplacian1(imgs, np.zeros(shape=(h, w)), 1e-5, 1).tocoo()
        indices = np.mat([coo.row, coo.col])
        indices = indices.transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)
