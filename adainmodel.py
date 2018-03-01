import tensorflow as tf
import tensorlayer.layers as tl
from vgg.custom_vgg19 import Vgg19
from model import Model
from layer.AdaINLayer import AdaINLayer
from layer.transposedconv2dlayer import TransposedConv2dLayer
import os
import utils

class AdaInModel(Model):
    def __init__(self,pretrained_vgg_path = None,adain_output_proportion = 1.0,content_loss_layer='conv4_1',style_loss_layer='conv1_1;conv2_1;conv3_1;conv4_1'):
        Model.__init__(self,channels=3)
        self.pretrained_vgg_path = pretrained_vgg_path
        self.adain_output_proportion = adain_output_proportion
        self.content_loss_layer = content_loss_layer
        self.style_loss_layers_list = style_loss_layer.split(';')
        print(self.style_loss_layers_list)

        assert os.path.exists(self.pretrained_vgg_path), 'The pretrained vgg file must exist!'


    def buildModel(self,isTrain=True):
        with tf.VariableScope('encoder'):
            encoder_input = tf.concat([self.content_input,self.style_input],0)
            encoder_output = Vgg19(encoder_input)
            encoder_output_content_loss_layer = getattr(encoder_output,self.content_loss_layer)
            encoder_content_output,encoder_style_output = tf.split(encoder_output_content_loss_layer,2,0)

        with tf.VariableScope('AdaIn-layer'):
            adain_content_input = tl.InputLayer(encoder_content_output,name='adain-content-input')
            adain_style_input = tl.InputLayer(encoder_style_output,name='adain-style-input')
            adain_output = AdaINLayer([adain_content_input,adain_style_input],self.adain_output_proportion,name='adain-layer')

        with tf.VariableScope('decoder'):
            decoder_layer_detail = utils.get_vgg19_decoder_layers_detail(self.content_loss_layer)

            decoder_input_channel = adain_output.outputs.get_shape()[-1].value
            decoder_layer_numbers = len(decoder_layer_detail)

            decoder_mild = tl.InputLayer(adain_output.outputs,name='decoder-input')
            for i in range(decoder_layer_numbers,0,-1):
                for j in range(decoder_layer_detail[i-1],0,-1):
                    decoder_mild = tl.Conv2d(decoder_mild,decoder_input_channel,[3,3],act=tf.nn.relu,name='conv%d_%d'%(i,j))
                if i!=1:
                    decoder_input_channel = decoder_input_channel // 2
                    decoder_mild = TransposedConv2dLayer(decoder_mild,decoder_input_channel,[3,3],[2,2],name='deconv%d'%(i-1))
                else:
                    decoder_mild = tl.Conv2d(decoder_mild,self.channels,[1,1],act=tf.nn.relu,name='conv%d_%d'%(0,0))

            self.outputs = decoder_mild.outputs




