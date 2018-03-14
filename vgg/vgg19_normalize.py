import tensorflow as tf
import utils
class NormalizeVgg19(object):

    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]
    def __init__(self,  input, weight = None, train=False):
        self.weight = weight

        print("building vgg19....")

        self.conv1_1 = self.conv_layer(input, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.avg_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.avg_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
        self.pool3 = self.avg_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
        self.pool4 = self.avg_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
        self.pool5 = self.avg_pool(self.conv5_4, 'pool5')


    def debug(self):
        pass

    def conv_layer(self,input,name):
        W_init = tf.constant_initializer(self.weight[name + '_W'])
        b_init = tf.constant_initializer(self.weight[name + '_b'])

        layer = tf.layers.conv2d(input,
                                 filters=utils.get_channel_number_from_vgg19_layer(name),
                                 name=name,
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_size=3,
                                 kernel_initializer=W_init,
                                 bias_initializer=b_init,
                                 trainable=False)
        return layer

    def avg_pool(self,input,name):
        layer = tf.layers.max_pooling2d(input,
                                        name=name,
                                        strides=2,
                                        pool_size=2)
        return layer

