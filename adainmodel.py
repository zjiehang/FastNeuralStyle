import tensorflow as tf
import tensorlayer.layers as tl
from vgg.vgg19_normalize import NormalizeVgg19
from model import Model
from layer.AdaINLayer import AdaINLayer
import shutil
import os
import utils
import time
from tqdm import tqdm
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
        self.use_gram = use_gram
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        self.encoder_input_channels = 3
        self.adain_input_channels = utils.get_channel_number_from_vgg19_layer(self.content_loss_layer)

        assert os.path.exists(self.pretrained_vgg_path), 'The pretrained vgg file must exist!'


    def buildTrainModel(self):
        # input for adain-lyaer
        self.content_input = tf.placeholder(tf.float32,[None,None,None,self.adain_input_channels],name='content-input')
        self.style_input = tf.placeholder(tf.float32,[None,None,None,self.adain_input_channels],name='style-input')


        with tf.variable_scope('AdaIn-layer'):
            self.transform_output = self.buildAdainLayer(self.content_input,self.style_input)

        with tf.variable_scope('decoder'):
            self.images = self.buildDecoder(self.transform_output,self.adain_input_channels)

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

        with tf.variable_scope('adain-layer'):
            self.transform_output = self.buildAdainLayer(self.content,self.style)

        with tf.variable_scope('decoder'):
            self.decoder_output = self.buildDecoder(self.transform_output,self.adain_input_channels)

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())


    def buildAdainLayer(self,content,style):
        adain_content_input_tl = tl.InputLayer(content, name='adain-content-input-tl')
        adain_style_input_tl = tl.InputLayer(style, name='adain-style-input-tl')
        adain_output_layer = AdaINLayer([adain_content_input_tl, adain_style_input_tl],
                                        self.adain_output_proportion,
                                        name='adain-layer')

        return tf.identity(adain_output_layer.outputs, name='transform-output')

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
                self.tv_loss = tf.squeeze(self.calculate_tv_loss(self.tv_loss_weight,self.images))

        self.all_loss = self.content_loss + self.style_loss + self.tv_loss


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


    def summaryMerge(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('content-loss',self.content_loss)
            tf.summary.scalar('style-loss',self.style_loss)
            tf.summary.scalar('tv-loss',self.tv_loss)
            tf.summary.scalar('all-loss',self.all_loss)
        self.summary_op = tf.summary.merge_all()


    """
    Train the neural network
    """
    def train(self,batch_size= 10, iterations=1000,save_dir="saved_models",reuse=False,reuse_dir=None,log_dir="log",summary_iter=100,save_iter=1000):

        #create the save directory if not exist
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.mkdir(log_dir)
        #Make new save directory
        os.mkdir(save_dir)
        #Operation to initialize all variables
        init = tf.global_variables_initializer()
        print("Begin training...")
        with self.sess as sess:
            #Initialize all variables
            sess.run(init)
            if reuse:
                self.resume(reuse_dir)
            #create summary writer for train
            writer = tf.summary.FileWriter(log_dir,sess.graph)

            #This is our training loop
            for i in tqdm(range(iterations)):
                start = time.time()
                content,style = self.data.get_batch(batch_size)

                # step 1
                # encode content and style images
                # input : content / style images
                # output : the vgg19 encoded version of content / style image

                content_batch_encoded = self.sess.run(self.encoder_content_output,feed_dict={self.images:content})
                style_batch_encoded,style_target_value = self.sess.run([self.encoder_content_output,self.encoder_style_output]
                                                                  ,feed_dict={self.images:style})


                # step 2
                # calculate the loss and run the train operation
                fetches = {
                    'train': self.train_op,
                    'global_step': self.global_step,
                    'summary': self.summary_op,
                    'lr': self.learning_rate,
                    'all_loss': self.all_loss,
                    'content_loss': self.content_loss,
                    'style_loss': self.style_loss,
                    'tv_loss': self.tv_loss
                }

                feed_dict = {
                    self.content_input:content_batch_encoded,
                    self.style_input:style_batch_encoded,
                    self.content_target:content_batch_encoded
                }
                for layer in self.style_loss_layers_list:
                    feed_dict[self.style_target[layer]] = style_target_value[layer]

                result = sess.run(fetches, feed_dict=feed_dict)

                ### Log the summaries
                if i % summary_iter == 0:
                    writer.add_summary(result['summary'], result['global_step'])

                ### Save checkpoint
                if i % save_iter == 0:
                    self.save(save_dir,result['global_step'])

                print("Step: {}  LR: {:.7f}  Loss: {:.5f}  Content: {:.5f}  Style: {:.5f}  tv: {:.5f}  Time: {:.5f}".format(
                    result['global_step'], result['lr'], result['all_loss'], result['content_loss'],result['style_loss'],result['tv_loss'],time.time() - start))
                # Last save
            self.save(save_dir, result['global_step'])
            writer.close()