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
import scipy.misc
from model.edsrmodel import EDSRModel
from utils.coral import coral_numpy

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
FLAGS = None

weight_lists = [
[1.0,0,0,0],
[0.75,0.25,0,0],
[0.5,0.5,0,0],
[0.25,0.75,0,0],
[0,1.0,0,0],
[0.75,0,0.25,0],
[0.5625,0.1875,0.1875,0.0625],
[0.375,0.375,0.125,0.125],
[0.1875,0.5625,0.0625,0.1875],
[0,0.75,0,0.5],
[0.5,0,0.5,0],
[0.375,0.125,0.375,0.125],
[0.25,0.25,0.25,0.25],
[0.125,0.375,0.125,0.375],
[0,0.5,0,0.5],
[0.25,0,0.75,0],
[0.1875,0.0625,0.5625,0.1875],
[0.125,0.125,0.375,0.375],
[0.0625,0.1875,0.1875,0.5625],
[0,0.25,0,0.75],
[0,0,1.0,0],
[0,0,0.75,0.25],
[0,0,0.5,0.5],
[0,0,0.25,0.75],
[0,0,0,1.0]
]

def main(_):
    if not os.path.exists(FLAGS.outdir):
        os.mkdir(FLAGS.outdir)
    _,content_postfix =  os.path.splitext(FLAGS.contentpath)
    content_batch = [scipy.misc.imread(FLAGS.contentpath,mode='RGB').astype(np.float32)/255.0]
    style_files = os.listdir(FLAGS.stylepath)
    style_batch = []
    print(style_files)
    for file in style_files:
        style_batch.append(scipy.misc.imread(FLAGS.stylepath+'/'+file,mode='RGB').astype(np.float32)/255.0)
    network = EDSRModel(FLAGS.pretrainedpath,
                         FLAGS.adainoutputproportion,
                         FLAGS.edsrlayer,
                         FLAGS.edsrfeaturesize,)
    network.buildMultiStylePredictModel()
    network.resume(FLAGS.reusedir)

    print('There are %d content pictures !'%(len(content_batch)))
    print('Thers are %d style pictures !'%(len(style_batch)))

    result_numbers = len(content_batch) * len(style_batch)
    start_time = time.time()

    for i in range(len(weight_lists)):
        each_time = time.time()
        output = network.predictMultiStyle(content_batch,style_batch,weight_lists[i])[0]
        output = np.clip(output,0.0,1.0)
        tl.vis.save_image(output,FLAGS.outdir + '/' + str(i//5) + '_' + str(i%5) +  content_postfix)
        print('Successfully saved %s in %.5f s'%((FLAGS.outdir + '/' + str(i//5) + '_' + str(i%5) +  content_postfix),float(time.time()-each_time)))

    print('Successfully process %d pictures!' % (result_numbers))
    print("Avg Time: %.5f /pic"%(float(time.time()-start_time)/result_numbers))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--edsrlayer",default=8,type=int)
    parser.add_argument("--edsrfeaturesize", default=128, type=int)

    parser.add_argument("--contentpath",default="/home/lkh/桌面/FastNeuralStyle/input/content/avril.jpg",type=str)
    parser.add_argument("--stylepath",default="/home/lkh/桌面/style_test",type=str)
    parser.add_argument("--pretrainedpath",default="pretrained/vgg19_weights_normalized.h5",type=str)
    parser.add_argument("--preservecolor",default=False,type=bool)
    #parser.add_argument("--stylelosslayers",default="conv1_1;conv2_1;conv3_1;conv4_1",type=str)
    parser.add_argument("--adainoutputproportion",default=1.0,type=float)
    parser.add_argument("--outdir",default='/home/lkh/桌面/multi_style_outfile',type=str)
    parser.add_argument("--reusedir", default='ckpt/edsr_content1_style1e-2_50000', type=str)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
