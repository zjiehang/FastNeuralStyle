import tensorflow as tf
import tensorlayer.layers as tl
from vgg.vgg19_normalize import NormalizeVgg19
from model import Model
from layer.AdaINLayer import AdaINLayer
from layer.UnpoolLayer import UnpoolLayer
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
                 use_affine = False,
                 affine_loss_weight = 0.0,
                 use_gram = False,
                 batch_size = 10,
                 learning_rate = 1e-4,
                 learning_rate_decay = 5e-5):
        self.pretrained_vgg_path = pretrained_vgg_path
        self.adain_output_proportion = adain_output_proportion
        self.content_loss_layer = content_loss_layer
        self.style_loss_layers_list = style_loss_layer.split(';')
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight
        self.tv_loss_weight = tv_loss_weight
        self.use_affine = use_affine
        self.affine_loss_weight = affine_loss_weight
        self.use_gram = use_gram
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        self.encoder_input_channels = 3
        self.adain_input_channels = utils.get_channel_number_from_vgg19_layer(self.content_loss_layer)

        assert os.path.exists(self.pretrained_vgg_path), 'The pretrained vgg file must exist!'


    def buildTrainModel(self):
        # input for adain-lyaer
        self.adain_content_input = tf.placeholder(tf.float32,[None,None,None,self.adain_input_channels],name='adain-content-input')
        self.adain_style_input = tf.placeholder(tf.float32,[None,None,None,self.adain_input_channels],name='adain-style-input')

        with tf.variable_scope('AdaIn-layer'):
            self.adain_output = self.buildAdainLayer(self.adain_content_input,self.adain_style_input)

        with tf.variable_scope('decoder'):
            self.images = self.buildDecoder(self.adain_output)

        with tf.variable_scope('encoder'):
            with open_weights(self.pretrained_vgg_path) as weight:
                vgg19 = NormalizeVgg19(self.images,weight)
            self.encoder_content_output = getattr(vgg19,self.content_loss_layer)
            self.encoder_style_output = {layer:getattr(vgg19,layer) for layer in self.style_loss_layers_list}

        self.calculateLoss()
        self.buildOptimizer()
        self.summaryMerge()

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.trainable_variables())

    def buildPredictModel(self):
        self.image = tf.placeholder(tf.float32,[None,None,None,3],name='image')
        self.content = tf.placeholder(tf.float32,[None,None,None,self.adain_input_channels],name='content')
        self.style = tf.placeholder(tf.float32,[None,None,None,self.adain_input_channels],name='style')

        with tf.variable_scope('encoder'):
            with open_weights(self.pretrained_vgg_path) as weight:
                vgg19 = NormalizeVgg19(self.image,weight)
            self.encoder_output = getattr(vgg19,self.content_loss_layer)

        with tf.variable_scope('AdaIn-layer'):
            self.adain_output = self.buildAdainLayer(self.content,self.style)

        with tf.variable_scope('decoder'):
            self.decoder_output = self.buildDecoder(self.adain_output)

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    def buildAdainLayer(self,content,style):
        adain_content_input_tl = tl.InputLayer(content, name='adain-content-input-tl')
        adain_style_input_tl = tl.InputLayer(style, name='adain-style-input-tl')
        adain_output_layer = AdaINLayer([adain_content_input_tl, adain_style_input_tl],
                                        self.adain_output_proportion,
                                        name='adain-layer')

        return tf.identity(adain_output_layer.outputs, name='adain-output')

    def buildDecoder(self,adain_output):
        decoder_middle = tl.InputLayer(adain_output, name='decoder-input')

        channels = self.adain_input_channels
        decoder_layer_detail = utils.get_vgg19_decoder_layers_detail(self.content_loss_layer)
        decoder_layer_numbers = len(decoder_layer_detail)

        for i in range(decoder_layer_numbers, 0, -1):
            for j in range(decoder_layer_detail[i - 1], 1, -1):
                decoder_middle = tl.Conv2d(decoder_middle,
                                           n_filter=channels,
                                           filter_size=[3, 3],
                                           act=tf.nn.relu,
                                           name='conv%d_%d' % (i, j))
            channels = channels // 2
            if i != 1:
                decoder_middle = tl.Conv2d(decoder_middle,
                                           n_filter=channels,
                                           filter_size=[3, 3],
                                           act=tf.nn.relu,
                                           name='conv%d_%d' % (i, 1))
                decoder_middle = UnpoolLayer(decoder_middle, scale=2, name='unpool%d' % (i - 1))
            else:
                decoder_middle = tl.Conv2d(decoder_middle,
                                           n_filter=3,
                                           filter_size=[3, 3],
                                           act=tf.nn.relu,
                                           name='conv%d_%d' % (i, 1))

        return tf.identity(decoder_middle.outputs,name='decoder-output')


    def calculateLoss(self):
        self.content_target = tf.placeholder(tf.float32,shape=[None,None,None,self.adain_input_channels])
        self.style_target = {
            layer:tf.placeholder(tf.float32,shape=[None,None,None,utils.get_channel_number_from_vgg19_layer(layer)])
            for layer in self.style_loss_layers_list
        }
        if self.use_affine:
            self.sparse_tensor_list = list()
            for i in range(self.batch_size):
                self.sparse_tensor_list.append(tf.sparse_placeholder(tf.float32))

        with tf.name_scope("loss"):
            with tf.name_scope("content-loss"):
                self.content_loss = tf.squeeze(self.calculate_content_loss(self.content_loss_weight,self.encoder_content_output,self.content_target))
            with tf.name_scope("style-loss"):
                self.style_loss = tf.squeeze(self.calculate_style_loss(self.style_loss_weight,self.use_gram,self.batch_size,self.encoder_style_output,self.style_target))
            with tf.name_scope("tv-loss"):
                self.tv_loss = tf.squeeze(self.calculate_tv_loss(self.tv_loss_weight,self.images))
            if self.use_affine:
                with tf.name_scope("affine-loss"):
                    self.affine_loss = self.calculate_affine_loss(self.affine_loss_weight,self.images,self.sparse_tensor_list)
        if self.use_affine:
            self.all_loss = self.content_loss + self.style_loss + self.tv_loss + self.affine_loss
        else:
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
            if self.use_affine:
                tf.summary.scalar('affine-loss', self.affine_loss)
            tf.summary.scalar('all-loss',self.all_loss)
        self.summary_op = tf.summary.merge_all()


    def calculate_content_loss(self,weight,x,y):
        return weight * utils.mean_squared(x,y)

    def calculate_style_loss(self,weight,use_gram,batch_size,x,y,epsilon=1e-5):
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

                m_loss = tf.reduce_sum(tf.squared_difference(smean, dmean)) / batch_size
                v_loss = tf.reduce_sum(tf.squared_difference(tf.sqrt(svar + epsilon), tf.sqrt(dvar + epsilon))) / batch_size

                style_loss_list.append(m_loss + v_loss)

            style_loss = weight * tf.reduce_sum(style_loss_list)
        return style_loss


    def calculate_tv_loss(self,weight,output):
        return tf.multiply(tf.reduce_mean(tf.image.total_variation(output)),weight)

    def calculate_affine_loss(self,weight,output,matting):
        loss_affine = 0.0
        output_split = tf.split(output,axis=0,num_or_size_splits=self.batch_size)

        for output_each,matting_each in zip(output_split,matting):
            for Vc in tf.unstack(output_each, axis=-1):
                Vc_ravel = tf.reshape(tf.transpose(Vc), [-1])
                loss_affine += tf.matmul(tf.expand_dims(Vc_ravel, 0),
                                     tf.sparse_tensor_dense_matmul(matting_each, tf.expand_dims(Vc_ravel, -1)))

        return loss_affine * weight /self.batch_size