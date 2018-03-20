import tensorlayer.layers as tl
import tensorflow as tf

class WCTLayer(tl.Layer):
    def __init__(self,layer=[],proportion = 1.0,epsilon=1e-5,eps=1e-8,batch_size=8,name ='wct-layer'):
        # check layer name (fixed)
        tl.Layer.__init__(self, name=name)
        print("  [TL] WCTLayer %s " %self.name)
        # the input of this layer is the output of previous layer (fixed)

        content_list = tf.split(layer[0].outputs,axis=0,num_or_size_splits=batch_size)
        style_list = tf.split(layer[1].outputs,axis=0,num_or_size_splits=batch_size)

        self.outputs = None

        for content,style in zip(content_list,style_list):
            '''
            TensorFlow version of Whiten-Color Transform
            Assume that content/style encodings have shape 1xHxWxC
            See p.4 of the Universal Style Transfer paper for corresponding equations:
            https://arxiv.org/pdf/1705.08086.pdf
            '''
            # Remove batch dim and reorder to CxHxW
            content_t = tf.transpose(tf.squeeze(content), (2, 0, 1))
            style_t = tf.transpose(tf.squeeze(style), (2, 0, 1))

            Cc, Hc, Wc = tf.unstack(tf.shape(content_t))
            Cs, Hs, Ws = tf.unstack(tf.shape(style_t))

            # CxHxW -> CxH*W
            content_flat = tf.reshape(content_t, (Cc, Hc * Wc))
            style_flat = tf.reshape(style_t, (Cs, Hs * Ws))

            # Content covariance
            mc = tf.reduce_mean(content_flat, axis=1, keep_dims=True)
            fc = content_flat - mc
            fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc * Wc, tf.float32) - 1.) + tf.eye(Cc) * eps

            # Style covariance
            ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
            fs = style_flat - ms
            fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs * Ws, tf.float32) - 1.) + tf.eye(Cs) * eps

            # tf.svd is slower on GPU, see https://github.com/tensorflow/tensorflow/issues/13603
            with tf.device('/cpu:0'):
                Sc, Uc, _ = tf.svd(fcfc)
                Ss, Us, _ = tf.svd(fsfs)

            ## Uncomment to perform SVD for content/style with np in one call
            ## This is slower than CPU tf.svd but won't segfault for ill-conditioned matrices
            # @jit
            # def np_svd(content, style):
            #     '''tf.py_func helper to run SVD with NumPy for content/style cov tensors'''
            #     Uc, Sc, _ = np.linalg.svd(content)
            #     Us, Ss, _ = np.linalg.svd(style)
            #     return Uc, Sc, Us, Ss
            # Uc, Sc, Us, Ss = tf.py_func(np_svd, [fcfc, fsfs], [tf.float32, tf.float32, tf.float32, tf.float32])

            # Filter small singular values
            k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, epsilon), tf.int32))
            k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, epsilon), tf.int32))

            # Whiten content feature
            Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))
            fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:, :k_c], Dc), Uc[:, :k_c], transpose_b=True), fc)

            # Color content with style
            Ds = tf.diag(tf.pow(Ss[:k_s], 0.5))
            fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:, :k_s], Ds), Us[:, :k_s], transpose_b=True), fc_hat)

            # Re-center with mean of style
            fcs_hat = fcs_hat + ms

            # Blend whiten-colored feature with original content feature
            blended = proportion * fcs_hat + (1 - proportion) * (fc + mc)

            # CxH*W -> CxHxW
            blended = tf.reshape(blended, (Cc, Hc, Wc))
            # CxHxW -> 1xHxWxC
            blended = tf.expand_dims(tf.transpose(blended, (1, 2, 0)), 0)
            if self.outputs is None:
                self.outputs = blended
            else:
                self.outputs = tf.concat([self.outputs,blended],axis=0)

        self.all_layers = list(layer[0].all_layers)
        self.all_params = list(layer[0].all_params)
        self.all_drop = dict(layer[0].all_drop)

        for i in range(1, len(layer)):
            self.all_layers.extend(list(layer[i].all_layers))
            self.all_params.extend(list(layer[i].all_params))
            self.all_drop.update(dict(layer[i].all_drop))

        self.all_layers = list_remove_repeat(self.all_layers)
        self.all_params = list_remove_repeat(self.all_params)


def list_remove_repeat(l=None):
    """Remove the repeated items in a list, and return the processed list.
    You may need it to create merged layer like Concat, Elementwise and etc.

    Parameters
    ----------
    l : a list

    Examples
    ---------
    > > > l = [2, 3, 4, 2, 3]
    > > > l = list_remove_repeat(l)
    ... [2, 3, 4]
    """
    l2 = []
    [l2.append(i) for i in l if not i in l2]
    return l2