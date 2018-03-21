# coding : utf-8
from __future__ import print_function
from __future__ import division
import sys
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import argparse
import os
import time
from wctmodel import WCTModel
from coral import coral_numpy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
FLAGS = None
def get_all_batch_from_path(path):
    assert os.path.exists(path),"The path must exists!"
    if os.path.isdir(path):
        return path,os.listdir(path)
    else:
        path_split = path.split('/')
        true_path = ''
        for i in range(len(path_split)-1):
            true_path = true_path + path_split[i] + '/'
        return true_path,[path_split[len(path_split)-1]]

def main(_):
    if not os.path.exists(FLAGS.outdir):
        os.mkdir(FLAGS.outdir)

    content_path,content_batch = get_all_batch_from_path(FLAGS.contentpath)
    style_path,style_batch = get_all_batch_from_path(FLAGS.stylepath)

    network = WCTModel(FLAGS.pretrainedpath,
                       FLAGS.wctoutputproportion,
                       FLAGS.contentlosslayer)
    network.buildPredictModel()
    network.resume(FLAGS.reusedir)

    print('There are %d content pictures !'%(len(content_batch)))
    print('Thers are %d style pictures !'%(len(style_batch)))

    result_numbers = len(content_batch) * len(style_batch)
    start_time = time.time()

    for content in content_batch:
        for style in style_batch:
            each_time = time.time()
            content_array = tl.visualize.read_image(content,content_path)
            style_array = tl.visualize.read_image(style,style_path)
            content_array = content_array.astype(np.float32)/255.0
            style_array = style_array.astype(np.float)/255.0

            if FLAGS.preservecolor:
                style_array = coral_numpy(style_array,content_array)

            content_name, content_post = os.path.splitext(content)
            style_name, style_post = os.path.splitext(style)
            output = network.predict([content_array],[style_array])[0]
            output = np.clip(output,0.0,1.0)
            tl.vis.save_image(output,FLAGS.outdir + '/' + content_name + '_stylized_' + style_name + content_post)
            print('Successfully saved %s in %.5f s'%((FLAGS.outdir + '/' + content_name + '_stylized_' + style_name + content_post),float(time.time()-each_time)))

    print('Successfully process %d pictures!' % (result_numbers))
    print("Avg Time: %.5f /pic"%(float(time.time()-start_time)/result_numbers))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--contentpath",default="input/content",type=str)
    parser.add_argument("--stylepath",default="input/style",type=str)
    parser.add_argument("--pretrainedpath",default="pretrained/vgg19_weights_normalized.h5",type=str)
    parser.add_argument("--contentlosslayer",default="conv5_1",type=str)
    parser.add_argument("--preservecolor",default=True,type=bool)
    #parser.add_argument("--stylelosslayers",default="conv1_1;conv2_1;conv3_1;conv4_1",type=str)
    parser.add_argument("--wctoutputproportion",default=1.0,type=float)
    parser.add_argument("--outdir",default='outfile/wct_10000',type=str)
    parser.add_argument("--reusedir", default='ckpt/wct_10000', type=str)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)