# coding : utf-8
import tensorlayer as tl
import os
import numpy as np
import utils

class Data(object):
    '''
    load training images including content images and style images
    '''
    def __init__(self,content_dir,style_dir,img_size = 256):
        self.content_dir = content_dir  #content images path
        self.style_dir = style_dir      #style images path

        self.img_size = img_size        #image size

        # check content directory
        assert os.path.exists(self.content_dir), 'The content image file must exist!'

        # check style directory
        assert os.path.exists(self.style_dir), 'The style image file must exist!'

        content_images = os.listdir(self.content_dir)
        style_images = os.listdir(self.style_dir)

        self.content_images = np.asarray(content_images)    #content images list
        self.style_images = np.asarray(style_images)        #style image list


    '''
    get batch of content images and style images
    '''
    def get_batch(self,batch_size = 8):
        # get content list
        content_batch_list = np.random.choice(self.content_images,size=batch_size)
        content_batch = self.__get_image_list(self.content_dir,content_batch_list)

        # get style list
        style_batch_list = np.random.choice(self.style_images)
        style_batch = self.__get_image_list(self.style_dir,style_batch_list)
        return content_batch,style_batch

    '''
    get picture list
    :param path: the path of images
    :param image_list: the list of images name to get 
    '''
    def __get_image_list(self,path,image_list):

        images_before_crop = tl.visualize.read_images(image_list,path)
        images_after_crop = []

        for i in range(len(images_before_crop)):
            image = utils.get_img_random_crop(images_before_crop[i], resize=self.img_size*2, crop=self.img_size)
            images_after_crop.append(image)

        return images_after_crop

