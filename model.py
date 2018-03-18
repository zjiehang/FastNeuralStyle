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

    """
    Train the neural network
    """
    def train(self,batch_size= 10, iterations=1000,save_dir="saved_models",reuse=False,reuse_dir=None,log_dir="log",summary_iter=100,save_iter=1000,use_affine=False):

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
                content,style,matpath = self.data.get_batch(batch_size)
                if use_affine:
                    sparse_tensor_value = utils.get_sqarse_tensor_value_by_mat(matpath,content[0].shape[0])


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
                    'tv_loss': self.tv_loss,
                    'affine_loss': self.affine_loss
                }
                feed_dict = {
                    self.adain_content_input:content_batch_encoded,
                    self.adain_style_input:style_batch_encoded,
                    self.content_target:content_batch_encoded
                }
                for layer in self.style_loss_layers_list:
                    feed_dict[self.style_target[layer]] = style_target_value[layer]
                if use_affine:
                    for j in range(len(sparse_tensor_value)):
                        feed_dict[self.sparse_tensor_list[j]] = sparse_tensor_value


                result = sess.run(fetches, feed_dict=feed_dict)

                ### Log the summaries
                if i % summary_iter == 0:
                    writer.add_summary(result['summary'], result['global_step'])

                ### Save checkpoint
                if i % save_iter == 0:
                    self.save(save_dir,result['global_step'])

                ### Debug
                print("Step: {}  LR: {:.7f}  Loss: {:.5f}  Content: {:.5f}  Style: {:.5f}  tv: {:.5f}  affine: {:.5f}  Time: {:.5f}".format(
                    result['global_step'], result['lr'], result['all_loss'], result['content_loss'],result['style_loss'],result['tv_loss'],result['affine_loss'],time.time() - start))

                # Last save
            self.save(save_dir, result['global_step'])
            writer.close()