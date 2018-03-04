import tensorflow as tf
from tqdm import tqdm
from abc import ABCMeta,abstractmethod
import os
import shutil
import time
import utils


"""
An implementation of the neural network used for
super-resolution of images as described in:

`Enhanced Deep Residual Networks for Single Image Super-Resolution`
(https://arxiv.org/pdf/1707.02921.pdf)

(single scale baseline-style model)
"""
class Model(object, metaclass=ABCMeta):
    def __init__(self,channels=3):
        self.channels = channels
        #Placeholder for image inputs
        self.content_input = tf.placeholder(tf.float32, [None, None, None, self.channels], name='content-input')
        #Placeholder for upscaled image ground-truth
        self.style_input = tf.placeholder(tf.float32, [None, None, None, self.channels], name='style-input')

        self.content_input_norm = utils.normalize_color(self.content_input)
        self.style_input_norm = utils.normalize_color(self.style_input)

    @abstractmethod
    def buildModel(self):
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
    def predict(self, x):
        return self.sess.run(self.output, feed_dict={self.input: x})

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
                feed_dict = {
                    self.content_input:content,
                    self.style_input:style
                }
                fetches = {
                    'train': self.train_op,
                    'global_step': self.global_step,
                    'summary': self.summary_op,
                    'lr': self.learning_rate,
                    'all_loss': self.all_loss,
                    'content_loss': self.content_loss,
                    'style_loss': self.style_loss,
                    'tv_loss': self.tv_loss,
                }
                result = sess.run(fetches, feed_dict=feed_dict)

                ### Log the summaries
                if i % summary_iter == 0:
                    writer.add_summary(result['summary'], result['global_step'])

                ### Save checkpoint
                if i % save_iter == 0:
                    self.save(save_dir,result['global_step'])

                ### Debug
                print("Step: {}  LR: {:.7f}  Loss: {:.5f}  Content: {:.5f}  Style: {:.5f}  tv: {:.5f}  Time: {:.5f}".format(
                    result['global_step'], result['lr'], result['all_loss'], result['content_loss'],result['style_loss'],result['tv_loss'],time.time() - start))

                # Last save
            self.save(save_dir, result['global_step'])
            writer.close()