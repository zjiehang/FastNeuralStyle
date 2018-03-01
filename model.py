import tensorflow as tf
from tqdm import tqdm
from abc import ABCMeta,abstractmethod
import os
import shutil


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


    @abstractmethod
    def buildModel(self):
        pass

    """
    Save the current state of the network to file
    """
    def save(self, savedir='saved_models'):
        print("Saving...")
        # tl.files.save_npz(self.all_params, name=savedir + '/model.npz', sess=self.sess)
        self.saver.save(self.sess,savedir+"/model")
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
    def train(self,batch_size= 10, iterations=1000,save_dir="saved_models",reuse=False,reuse_dir=None,log_dir="log"):
        if reuse:
            self.resume(reuse_dir)

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

            #create summary writer for train
            train_writer = tf.summary.FileWriter(log_dir+"/train",sess.graph)

            #If we're using a test set, include another summary writer for that
            test_writer = tf.summary.FileWriter(log_dir+"/test",sess.graph)
            test_feed = []
            while True:
                test_x,test_y = self.data.get_test_set(batch_size)
                if test_x!=None and test_y!=None:
                    test_feed.append({
                            self.input:test_x,
                            self.target:test_y
                    })
                else:
                    break

            #This is our training loop
            for i in tqdm(range(iterations)):
                #Use the data function we were passed to get a batch every iteration
                x,y = self.data.get_batch(batch_size)
                #Create feed dictionary for the batch
                feed = {
                    self.input:x,
                    self.target:y
                }
                #Run the train op and calculate the train summary
                summary,_ = sess.run([self.train_merge,self.train_op],feed)
                #Write train summary for this step
                if i%10 == 0:
                    train_writer.add_summary(summary,i)
                #test every 10 iterations
                if i%100 == 0:
                    sess.run(tf.local_variables_initializer())
                    for j in range(len(test_feed)):
                        sess.run([self.streaming_loss_update,self.streaming_psnr_update],feed_dict=test_feed[j])
                    streaming_summ = sess.run(self.test_merge)
                    #Write test summary
                    test_writer.add_summary(streaming_summ,i)

                # Save our trained model
                if i!=0 and i % 500 == 0:
                    self.save(save_dir)

            self.save(save_dir)
            test_writer.close()
            train_writer.close()