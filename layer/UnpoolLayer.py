import tensorlayer.layers as tl
import tensorflow as tf

class UnpoolLayer(tl.Layer):
    def __init__(self,layer=None,scale=2,name ='unpool-layer'):
        # check layer name (fixed)
        tl.Layer.__init__(self, name=name)

        self.inputs = layer.outputs

        print("  [TL] UnpoolLayer %s " %self.name)
        # the input of this layer is the output of previous layer (fixed)

        shape = tf.shape(self.inputs)
        h,w = shape[1],shape[2]
        self.outputs = tf.image.resize_nearest_neighbor(self.inputs,tf.stack([h*scale,w*scale]))

        # get stuff from previous layer (fixed)
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        # update layer (customized)
        self.all_layers.extend([self.outputs])

