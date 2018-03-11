# coding : utf-8
from __future__ import print_function
from __future__ import division
import sys
import tensorflow as tf
import tensorlayer as tl
import scipy.misc
import argparse
import os
import time
from adainmodel import AdaInModel
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
FLAGS = None
def get_all_batch_from_path(path):
    assert os.path.exists(path),"The path must exists!"
    if os.path.isdir(path):
        files = os.path.join(path)
        return [os.path.join(path,file) for file in files]
    else:
        return [path]

def main(_):
    if not os.path.exists(FLAGS.outdir):
        os.mkdir(FLAGS.outdir)

    content_batch = get_all_batch_from_path(FLAGS.contentpath)
    style_batch = get_all_batch_from_path(FLAGS.stylepath)

    network = AdaInModel(FLAGS.pretrainedpath,
                         FLAGS.adainoutputproportion,
                         FLAGS.contentlosslayer,
                         FLAGS.stylelosslayers)
    network.buildModel(isTrain=False)

    print('There are %d content pictures !'%(len(content_batch)))
    print('Thers are %d style pictures !'%(len(style_batch)))

    result_numbers = len(content_batch) * len(style_batch)
    start_time = time.time()

    for content in content_batch:
        for style in style_batch:
            content_array = scipy.misc.imread(content)
            style_array = scipy.misc.imread(style)
            content_name, content_post = os.path.splitext(content)
            style_name, style_post = os.path.splitext(style)
            output = network.predict([content_array],[style_array])
            tl.vis.save_image(output[0],FLAGS.outdir + '/' + content_name + '_stylized_' + style_name + content_post)

    print('Successfully process %d pictures!' % (result_numbers))
    print("Avg Time: %.5f /pic"%(float(time.time()-start_time)/result_numbers))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--contentpath",default="images/content",type=str)
    parser.add_argument("--stylepath",default="images/style",type=str)
    parser.add_argument("--pretrainedpath",default="pretrained/vgg19.npy",type=str)
    parser.add_argument("--contentlosslayer",default="conv4_1",type=str)
    #parser.add_argument("--stylelosslayers",default="conv1_1;conv2_1;conv3_1;conv4_1",type=str)
    parser.add_argument("--adainoutputproportion",default=1.0,type=float)
    parser.add_argument("--logdir",default='logfile/adain_10000',type=str)
    parser.add_argument("--reusedir", default='ckpt/adain_10000', type=str)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)