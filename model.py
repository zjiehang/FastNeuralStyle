import tensorflow as tf
from tqdm import tqdm
from abc import ABCMeta,abstractmethod
import os
import utils
import tensorlayer.layers as tl
from layer.UnpoolLayer import UnpoolLayer


"""
An implementation of the neural network used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""
class Model(object, metaclass=ABCMeta):
    @abstractmethod
    def buildTrainModel(self):
        pass

    @abstractmethod
    def buildPredictModel(self):
        pass

    """
    Save the current state of the network to file
    """
    def save(self, savedir='saved_models',global_step = 0):
        print("Saving...")
        # tl.files.save_npz(self.all_params, name=savedir + '/model.npz', sess=self.sess)
        self.saver.save(self.sess,savedir+"/model",global_step)
        print("Saved!")

    """
    Resume network from previously saved weights
    """
    def resume(self,savedir='saved_models'):
        if os.path.exists(savedir):
            print("Restoring...")
            self.saver.restore(self.sess,tf.train.latest_checkpoint(savedir))
            print("Restored!")

    """
    Function to setup your input data pipeline
    """
    def set_data(self, data):
        self.data = data


    """
    Estimate the trained model
    x: (tf.float32, [batch_size, h, w, output_channels])
    """
    def predict(self, content,style):
        content_encoded = self.sess.run(self.encoder_output,feed_dict={self.image:content})
        style_encoded = self.sess.run(self.encoder_output,feed_dict={self.image:style})

        return self.sess.run(self.decoder_output,feed_dict={self.content:content_encoded,
                                                    self.style:style_encoded})


    def buildDecoder(self,adain_output,input_channels):
        decoder_middle = tl.InputLayer(adain_output, name='decoder-input')

        channels = input_channels
        decoder_layer_detail = utils.get_vgg19_decoder_layers_detail(self.content_loss_layer)
        decoder_layer_numbers = len(decoder_layer_detail)

        for i in range(decoder_layer_numbers, 0, -1):
            for j in range(decoder_layer_detail[i - 1], 1, -1):
                decoder_middle = tl.Conv2dLayer(decoder_middle,
                                                shape=[3,3,channels,channels],
                                                strides=[1,1,1,1],
                                                act=tf.nn.relu,
                                                name='conv%d_%d' % (i, j))
            if i == 5:
                decoder_middle = tl.Conv2dLayer(decoder_middle,
                                                shape=[3,3,channels,channels],
                                                strides=[1,1,1,1],
                                                act=tf.nn.relu,
                                                name='conv%d_%d' % (i, 1))
                decoder_middle = UnpoolLayer(decoder_middle, scale=2, name='unpool%d' % (i - 1))
            elif i==1:
                decoder_middle = tl.Conv2dLayer(decoder_middle,
                                                shape=[3,3,channels,3],
                                                strides=[1,1,1,1],
                                                act=tf.nn.relu,
                                                name='conv%d_%d' % (i, 1))
            else:
                channels = channels // 2
                decoder_middle = tl.Conv2dLayer(decoder_middle,
                                                shape=[3, 3, channels * 2, channels],
                                                strides=[1, 1, 1, 1],
                                                act=tf.nn.relu,
                                                name='conv%d_%d' % (i, 1))
                decoder_middle = UnpoolLayer(decoder_middle, scale=2, name='unpool%d' % (i - 1))


        return tf.identity(decoder_middle.outputs,name='decoder-output')

    def buildOptimizer(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate = tf.train.inverse_time_decay(self.learning_rate,
                                                         self.global_step,
                                                         decay_steps=1,
                                                         decay_rate=self.learning_rate_decay)

        # self.learning_rate = utils.learning_rate_decay(self.learning_rate,
        #                                               self.global_step,
        #                                               self.learning_rate_decay)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.all_loss, global_step=self.global_step)
