import tensorflow as tf
import tensorlayer.layers as tl
from vgg.vgg19_normalize import NormalizeVgg19
from model import Model
from layer.AdaINLayer import AdaINLayer
import os
import utils
from weights import open_weights


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

        self.encoder_input_channels = 3
        self.adain_input_channels = utils.get_vgg19_decoder_layers_detail(self.content_loss_layer)

        # input for encoder/vgg19
        self.encoder_input = tf.placeholder(tf.float32,[None,None,None,self.encoder_input_channels],name='encoder-input')

        # input for adain-lyaer
        self.adain_content_input = tf.placeholder(tf.float32,[None,None,None,self.adain_input_channels],name='adain-content-input')
        self.adain_style_input = tf.placeholder(tf.float32,[None,None,None,self.adain_input_channels],name='adain-style-input')

        # input for decoder
        self.decoder_input = tf.placeholder(tf.float32,[None,None,None,self.adain_input_channels],name='decoder-input')

        assert os.path.exists(self.pretrained_vgg_path), 'The pretrained vgg file must exist!'


    def buildModel(self,isTrain=True):
        with tf.variable_scope('encoder'):
            with open_weights(self.pretrained_vgg_path) as weight:
                vgg19 = NormalizeVgg19(self.encoder_input,weight)
            self.encoder_content_output = getattr(vgg19,self.content_loss_layer)
            self.encoder_style_output = {layer:getattr(vgg19,layer) for layer in self.style_loss_layers_list}

            self.encoder_output = tf.identity(self.encoder_content_output,name='encoder-output')
            '''
            self.vgg_encode_content_input = CustomVgg19(self.content_input_norm,self.pretrained_vgg_path)
            self.vgg_encode_style_input = CustomVgg19(self.style_input_norm,self.pretrained_vgg_path)
            encoder_content_output = getattr(self.vgg_encode_content_input,self.content_loss_layer)
            encoder_style_output = getattr(self.vgg_encode_style_input,self.content_loss_layer)
            '''

        with tf.variable_scope('AdaIn-layer'):
            adain_content_input_tl = tl.InputLayer(self.adain_content_input,name='adain-content-input-tl')
            adain_style_input_tl = tl.InputLayer(self.adain_style_input,name='adain-style-input-tl')
            adain_output = AdaINLayer([adain_content_input_tl,adain_style_input_tl],
                                           self.adain_output_proportion,
                                           name='adain-layer')
            self.adain_layer_output = tf.identity(adain_output.outputs,name='adain-output')

            '''
            adain_content_input = tl.InputLayer(encoder_content_output,name='adain-content-input')
            adain_style_input = tl.InputLayer(encoder_style_output,name='adain-style-input')
            self.adain_output = AdaINLayer([adain_content_input,adain_style_input],self.adain_output_proportion,name='adain-layer')
            '''

        with tf.variable_scope('decoder'):
            decoder_middle = tl.InputLayer(self.decoder_input,name='decoder-input-tl')

            channels = self.adain_input_channels
            decoder_layer_detail = utils.get_vgg19_decoder_layers_detail(self.content_loss_layer)
            decoder_layer_numbers = len(decoder_layer_detail)

            for i in range(decoder_layer_numbers,0,-1):
                for j in range(decoder_layer_detail[i-1],1,-1):
                    decoder_middle = tl.Conv2d(decoder_middle,
                                               n_filter=channels,
                                               filter_size=[3,3],
                                               act=tf.nn.relu,
                                               W_init=tf.contrib.layers.xavier_initializer,
                                               b_init=tf.zeros_initializer,
                                               name='conv%d_%d'%(i,j))
                channels = channels // 2
                if i!=1:
                    decoder_middle = tl.Conv2d(decoder_middle,
                                               n_filter=channels,
                                               filter_size=[3,3],
                                               act=tf.nn.relu,
                                               W_init=tf.contrib.layers.xavier_initializer,
                                               b_init=tf.zeros_initializer,
                                               name='conv%d_%d'%(i,1))
                    decoder_middle = tl.UpSampling2dLayer(decoder_middle,size=[2,2],is_scale=True,name='unpool%d'%(i-1))
                else:
                    decoder_middle = tl.Conv2d(decoder_middle,
                                               n_filter=3,
                                               filter_size=[3,3],
                                               act=tf.nn.relu,
                                               W_init=tf.contrib.layers.xavier_initializer,
                                               b_init=tf.zeros_initializer,
                                               name='conv%d_%d'%(i,1))

            self.decoder_output = tf.identity(decoder_middle.outputs,name='decoder-output')
            '''
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

            self.outputs = tf.clip_by_value(decoder_mild.outputs,0.0,1.0)
            '''
        if isTrain:
            self.calculateLoss()
            self.buildOptimizer()
            self.summaryMerge()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()


    def calculateLoss(self):
        self.content_target = tf.placeholder(tf.float32,shape=[None,None,None,self.adain_input_channels])
        self.style_target = {
            layer:tf.placeholder(tf.float32,shape=[None,None,None,utils.get_channel_number_from_vgg19_layer(layer)])
            for layer in self.style_loss_layers_list
        }

        with tf.name_scope("loss"):
            with tf.name_scope("content-loss"):
                self.content_loss = tf.squeeze(self.calculate_content_loss(self.content_loss_weight,self.encoder_content_output,self.content_target))
            with tf.name_scope("style-loss"):
                self.style_loss = tf.squeeze(self.calculate_style_loss(self.style_loss_weight,self.use_gram,self.batch_size,self.encoder_style_output,self.style_target))
            with tf.name_scope("tv-loss"):
                self.tv_loss = tf.squeeze(self.calculate_tv_loss(self.tv_loss_weight,self.encoder_input))
            #with tf.name_scope("realistic-loss"):
            #    self.realistic_loss = self.calculate_realistic_loss()
        #self.all_loss = self.content_loss + self.style_loss + self.tv_loss + self.realistic_loss
        self.all_loss = self.content_loss + self.style_loss + self.tv_loss

    def buildOptimizer(self):
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.learning_rate = tf.train.inverse_time_decay(self.learning_rate,
                                                         self.global_step,
                                                         decay_steps=1,
                                                         decay_rate=self.learning_rate_decay)

        #self.learning_rate = utils.learning_rate_decay(self.learning_rate,
        #                                               self.global_step,
        #                                               self.learning_rate_decay)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.all_loss,global_step=self.global_step)


    def summaryMerge(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('content-loss',self.content_loss)
            tf.summary.scalar('style-loss',self.style_loss)
            tf.summary.scalar('tv-loss',self.tv_loss)
            tf.summary.scalar('all-loss',self.all_loss)
            '''
            tf.summary.image('content-image',tf.clip_by_value(self.,0,1))
            tf.summary.image('style-image',tf.clip_by_value(self.style_input_norm,0,1))
            tf.summary.image('stylied-image',tf.clip_by_value(self.outputs,0,1))
            '''
        self.summary_op = tf.summary.merge_all()


    def calculate_content_loss(self,weight,x,y):
        return weight * utils.mean_squared(x,y)

    def calculate_style_loss(self,weight,use_gram,batch_size,x,y):
        if use_gram is True:
            gram_losses = []
            for layer in self.style_loss_layers_list:
                s_gram = utils.gram_matrix(x[layer])
                d_gram = utils.gram_matrix(y[layer])
                gram_loss = utils.mean_squared(d_gram, s_gram)
                gram_losses.append(gram_loss)
            style_loss = weight * tf.reduce_sum(gram_losses) / batch_size
        else:
            style_loss_list = []
            for layer in self.style_loss_layers_list:
                smean, svar = tf.nn.moments(x[layer], [1, 2])
                dmean, dvar = tf.nn.moments(y[layer], [1, 2])

                m_loss = utils.mean_squared(smean, dmean) / batch_size
                v_loss = utils.mean_squared(tf.sqrt(svar), tf.sqrt(dvar)) / batch_size

                style_loss_list.append(m_loss + v_loss)

            style_loss = weight * tf.reduce_sum(style_loss_list)
        return style_loss


    def calculate_tv_loss(self,weight,output):
        return tf.multiply(tf.reduce_mean(tf.image.total_variation(output)),weight)