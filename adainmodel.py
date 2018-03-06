import tensorflow as tf
import tensorlayer.layers as tl
from vgg.custom_vgg19 import CustomVgg19
from model import Model
from layer.AdaINLayer import AdaINLayer
from layer.transposedconv2dlayer import TransposedConv2dLayer
import os
import utils

class AdaInModel(Model):
    def __init__(self,
                 pretrained_vgg_path = None,
                 adain_output_proportion = 1.0,
                 content_loss_layer='conv4_1',
                 style_loss_layer='conv1_1;conv2_1;conv3_1;conv4_1',
                 content_loss_weight = 1.0,
                 style_loss_weight = 1.0,
                 tv_loss_weight = 0.0,
                 realistic_loss_weight = 0.0,
                 use_gram = False,
                 batch_size = 10,
                 learning_rate = 1e-4,
                 learning_rate_decay = 5e-5):
        Model.__init__(self,channels=3)

        self.pretrained_vgg_path = pretrained_vgg_path
        self.adain_output_proportion = adain_output_proportion
        self.content_loss_layer = content_loss_layer
        self.style_loss_layers_list = style_loss_layer.split(';')
        self.content_loss_weight = content_loss_weight,
        self.style_loss_weight = style_loss_weight,
        self.tv_loss_weight = tv_loss_weight,
        self.realistic_loss_weight = realistic_loss_weight,
        self.use_gram = use_gram
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        assert os.path.exists(self.pretrained_vgg_path), 'The pretrained vgg file must exist!'


    def buildModel(self,isTrain=True):
        with tf.variable_scope('encoder'):
            encoder_input = tf.concat([self.content_input_norm,self.style_input_norm],0)
            self.vgg_encode_input = CustomVgg19(encoder_input,self.pretrained_vgg_path)
            encoder_output_content_loss_layer = getattr(self.vgg_encode_input,self.content_loss_layer)
            encoder_content_output,encoder_style_output = tf.split(encoder_output_content_loss_layer,2,0)

        with tf.variable_scope('AdaIn-layer'):
            adain_content_input = tl.InputLayer(encoder_content_output,name='adain-content-input')
            adain_style_input = tl.InputLayer(encoder_style_output,name='adain-style-input')
            self.adain_output = AdaINLayer([adain_content_input,adain_style_input],self.adain_output_proportion,name='adain-layer')

        with tf.variable_scope('decoder'):
            decoder_layer_detail = utils.get_vgg19_decoder_layers_detail(self.content_loss_layer)

            decoder_input_channel = self.adain_output.outputs.get_shape()[-1].value
            decoder_layer_numbers = len(decoder_layer_detail)

            decoder_mild = tl.InputLayer(self.adain_output.outputs,name='decoder-input')
            for i in range(decoder_layer_numbers,0,-1):
                for j in range(decoder_layer_detail[i-1],0,-1):
                    decoder_mild = tl.Conv2d(decoder_mild,decoder_input_channel,[3,3],act=tf.nn.relu,name='conv%d_%d'%(i,j))
                if i!=1:
                    decoder_input_channel = decoder_input_channel // 2
                    decoder_mild = TransposedConv2dLayer(decoder_mild,decoder_input_channel,[3,3],[2,2],name='deconv%d'%(i-1))
                else:
                    decoder_mild = tl.Conv2d(decoder_mild,self.channels,[1,1],act=tf.nn.relu,name='conv%d_%d'%(0,0))

            self.outputs = decoder_mild.outputs

        if isTrain:
            self.calculateLoss()
            self.buildOptimizer()
            self.summaryMerge()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()


    def calculateLoss(self):
        vgg_encode_decoder_output = CustomVgg19(self.outputs,self.pretrained_vgg_path)

        '''
        use the feature responses of the content images
        '''
        # content_loss_input = tf.split(getattr(self.vgg_encode_input,self.content_loss_layer),2,0)[0]
        '''
        use the output of adain-layer as the content loss component
        '''
        content_loss_input = self.adain_output.outputs
        content_loss_output = getattr(vgg_encode_decoder_output,self.content_loss_layer)

        style_loss_input = []
        style_loss_output = []

        for layer in self.style_loss_layers_list:
            style_loss_input.append(tf.split(getattr(self.vgg_encode_input,layer),2,0)[1])
            style_loss_output.append(getattr(vgg_encode_decoder_output,layer))

        with tf.name_scope("loss"):
            with tf.name_scope("content-loss"):
                self.content_loss = tf.squeeze(self.calculate_content_loss(self.content_loss_weight,content_loss_input,content_loss_output))
            with tf.name_scope("style-loss"):
                self.style_loss = tf.squeeze(self.calculate_style_loss(self.style_loss_weight,self.use_gram,self.batch_size,style_loss_input,style_loss_output))
            with tf.name_scope("tv-loss"):
                self.tv_loss = tf.squeeze(self.calculate_tv_loss(self.tv_loss_weight,self.outputs))
            #with tf.name_scope("realistic-loss"):
            #    self.realistic_loss = self.calculate_realistic_loss()
        #self.all_loss = self.content_loss + self.style_loss + self.tv_loss + self.realistic_loss
        self.all_loss = self.content_loss + self.style_loss + self.tv_loss

    def buildOptimizer(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate = utils.learning_rate_decay(self.learning_rate,
                                                       self.global_step,
                                                       self.learning_rate_decay)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.all_loss,global_step=self.global_step)


    def summaryMerge(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('content-loss',self.content_loss)
            tf.summary.scalar('style-loss',self.style_loss)
            tf.summary.scalar('tv-loss',self.tv_loss)
            tf.summary.scalar('all-loss',self.all_loss)

            tf.summary.image('content-image',tf.clip_by_value(self.content_input_norm,0,1))
            tf.summary.image('style-image',tf.clip_by_value(self.style_input_norm,0,1))
            tf.summary.image('stylied-image',tf.clip_by_value(self.outputs,0,1))

        self.summary_op = tf.summary.merge_all()


    def calculate_content_loss(self,weight,x,y):
        return weight * utils.mean_squared(x,y)

    def calculate_style_loss(self,weight,use_gram,batch_size,x,y):
        if use_gram is True:
            gram_losses = []
            for smap, dmap in zip(x, y):
                s_gram = utils.gram_matrix(smap)
                d_gram = utils.gram_matrix(dmap)
                gram_loss = utils.mean_squared(d_gram, s_gram)
                gram_losses.append(gram_loss)
            style_loss = weight * tf.reduce_sum(gram_losses) / batch_size
        else:
            style_loss_list = []
            for smap, dmap in zip(x, y):
                smean, svar = tf.nn.moments(smap, [1, 2])
                dmean, dvar = tf.nn.moments(dmap, [1, 2])

                m_loss = utils.mean_squared(smean, dmean) / batch_size
                v_loss = utils.mean_squared(tf.sqrt(svar), tf.sqrt(dvar)) / batch_size

                style_loss_list.append(m_loss + v_loss)

            style_loss = weight * tf.reduce_sum(style_loss_list)
        return style_loss


    def calculate_tv_loss(self,weight,output):
        return tf.multiply(tf.reduce_mean(tf.image.total_variation(output)),weight)