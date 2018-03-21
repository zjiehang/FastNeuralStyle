import tensorflow as tf
import tensorlayer.layers as tl
from vgg.vgg19_normalize import NormalizeVgg19
from model import Model
from layer.WCTLayer import WCTLayer
import os
import utils
import shutil
from tqdm import tqdm
import time
from weights import open_weights


class WCTModel(Model):
    def __init__(self,
                 pretrained_vgg_path = None,
                 wct_output_proportion = 1.0,
                 content_loss_layer='conv5_1',
                 pixel_loss_weight = 1.0,
                 feature_loss_weight = 1.0,
                 tv_loss_weight = 0.0,
                 batch_size = 10,
                 learning_rate = 1e-4,
                 learning_rate_decay = 5e-5):
        self.pretrained_vgg_path = pretrained_vgg_path
        self.wct_output_proportion = wct_output_proportion
        self.content_loss_layer = content_loss_layer
        self.pixel_loss_weight = pixel_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.tv_loss_weight = tv_loss_weight
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        self.encoder_input_channels = 3
        self.wct_input_channels = utils.get_channel_number_from_vgg19_layer(self.content_loss_layer)

        assert os.path.exists(self.pretrained_vgg_path), 'The pretrained vgg file must exist!'


    def buildTrainModel(self):
        # input for wct-lyaer
        self.content_input = tf.placeholder(tf.float32,[None,None,None,self.wct_input_channels],name='content-input')
        self.style_input = tf.placeholder(tf.float32,[None,None,None,self.wct_input_channels],name='style-input')


        with tf.variable_scope('WCT-layer'):
            self.transform_output = self.buildWCTLayer(self.content_input,self.style_input)

        with tf.variable_scope('decoder'):
            self.images = self.buildDecoder(self.transform_output,self.wct_input_channels)

        with tf.variable_scope('encoder'):
            with open_weights(self.pretrained_vgg_path) as weight:
                vgg19 = NormalizeVgg19(self.images,weight)
            self.encoder_content_output = getattr(vgg19,self.content_loss_layer)

        self.calculateLoss()
        self.buildOptimizer()
        self.summaryMerge()

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.trainable_variables())

    def buildPredictModel(self):
        self.image = tf.placeholder(tf.float32,[None,None,None,3],name='image')
        self.content = tf.placeholder(tf.float32,[None,None,None,self.wct_input_channels],name='content')
        self.style = tf.placeholder(tf.float32,[None,None,None,self.wct_input_channels],name='style')

        with tf.variable_scope('encoder'):
            with open_weights(self.pretrained_vgg_path) as weight:
                vgg19 = NormalizeVgg19(self.image,weight)
            self.encoder_output = getattr(vgg19,self.content_loss_layer)

        with tf.variable_scope('WCT-layer'):
            self.transform_output = self.buildWCTLayer(self.content,self.style)

        with tf.variable_scope('decoder'):
            self.decoder_output = self.buildDecoder(self.transform_output)

        self.sess = tf.Session()
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess.run(tf.global_variables_initializer())

    def buildWCTLayer(self,content,style):
        wct_content_input_tl = tl.InputLayer(content, name='wct-content-input-tl')
        wct_style_input_tl = tl.InputLayer(style, name='wct-style-input-tl')
        wct_output_layer = WCTLayer([wct_content_input_tl, wct_style_input_tl],
                                        self.wct_output_proportion,
                                        batch_size=self.batch_size,
                                        name='wct-layer')

        return tf.identity(wct_output_layer.outputs, name='transform-output')


    def calculateLoss(self):
        self.content_target = tf.placeholder(tf.float32,shape=[None,None,None,self.wct_input_channels])
        self.content_image = tf.placeholder(tf.float32,shape=[None,None,None,3])

        with tf.name_scope("loss"):
            with tf.name_scope("feature-loss"):
                self.feature_loss = tf.squeeze(self.calculate_mse_loss(self.feature_loss_weight,self.encoder_content_output,self.content_target))
            with tf.name_scope("pixel-loss"):
                self.pixel_loss = tf.squeeze(self.calculate_mse_loss(self.pixel_loss_weight,self.images,self.content_image))
            with tf.name_scope("tv-loss"):
                self.tv_loss = tf.squeeze(self.calculate_tv_loss(self.tv_loss_weight,self.images))

        self.all_loss = self.feature_loss + self.pixel_loss + self.tv_loss

    def summaryMerge(self):
        with tf.name_scope('summary'):
            tf.summary.scalar('feature-loss',self.feature_loss)
            tf.summary.scalar('pixel-loss',self.pixel_loss)
            tf.summary.scalar('tv-loss',self.tv_loss)
            tf.summary.scalar('all-loss',self.all_loss)
        self.summary_op = tf.summary.merge_all()

    def calculate_mse_loss(self,weight,output,target):
        return weight*utils.mean_squared(output,target)

    def calculate_tv_loss(self,weight,output):
        return tf.multiply(tf.reduce_mean(tf.image.total_variation(output)),weight)

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
                style_batch_encoded = self.sess.run(self.encoder_content_output,feed_dict={self.images:style})

                # step 2
                # calculate the loss and run the train operation
                fetches = {
                    'train': self.train_op,
                    'global_step': self.global_step,
                    'summary': self.summary_op,
                    'lr': self.learning_rate,
                    'all_loss': self.all_loss,
                    'feature_loss': self.feature_loss,
                    'pixel_loss': self.pixel_loss,
                    'tv_loss': self.tv_loss
                }

                feed_dict = {
                    self.content_input:content_batch_encoded,
                    self.style_input:style_batch_encoded,
                    self.content_target:content_batch_encoded,
                    self.content_image:content
                }

                result = sess.run(fetches, feed_dict=feed_dict)

                ### Log the summaries
                if i % summary_iter == 0:
                    writer.add_summary(result['summary'], result['global_step'])

                ### Save checkpoint
                if i % save_iter == 0:
                    self.save(save_dir,result['global_step'])

                print("Step: {}  LR: {:.7f}  Loss: {:.5f}  Pixel: {:.5f}  Feature: {:.5f}  tv: {:.5f}  Time: {:.5f}".format(
                    result['global_step'], result['lr'], result['all_loss'], result['pixel_loss'],result['feature_loss'],result['tv_loss'],time.time() - start))
                # Last save
            self.save(save_dir, result['global_step'])
            writer.close()