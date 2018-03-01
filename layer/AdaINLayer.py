import tensorlayer.layers as tl
import tensorflow as tf

class AdaINLayer(tl.Layer):
    def __init__(self,layer=[],proportion = 1.0,epsilon=1e-5,name ='adain-layer'):
        # check layer name (fixed)
        tl.Layer.__init__(self, name=name)
        print("  [TL] AdaINLayer %s " %self.name)
        # the input of this layer is the output of previous layer (fixed)

        layer0_mean,layer0_variance = tf.nn.moments(layer[0].outputs,[1,2],keep_dims=True)
        layer1_mean,layer1_variance = tf.nn.moments(layer[1].outputs,[1,2],keep_dims=True)

        normanized = tf.nn.batch_normalization(layer[0].outputs,
                                               layer0_mean,
                                               layer0_variance,
                                               layer1_mean,
                                               tf.sqrt(layer1_variance),
                                               variance_epsilon=epsilon)

        self.outputs = normanized * proportion + layer[0].outputs * (1.0 - proportion)

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