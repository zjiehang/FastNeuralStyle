# coding : utf-8
from __future__ import print_function
from __future__ import division
import sys
import tensorflow as tf
import argparse
import os
from data import Data
from adainmodel import AdaInModel

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
FLAGS = None

def main(_):
    #data = Data(FLAGS.contentpath,FLAGS.stylepath,FLAGS.imgsize)
    network = AdaInModel(FLAGS.pretrainedpath,FLAGS.adainoutputproportion,FLAGS.contentlosslayer,FLAGS.stylelosslayers)
    network.buildModel(isTrain=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--contentpath",default="images/content",type=str)
    parser.add_argument("--stylepath",default="images/style",type=str)
    parser.add_argument("--pretrainedpath",default="pretrained/vgg19.npy",type=str)
    parser.add_argument("--contentlosslayer",default="conv4_1",type=str)
    parser.add_argument("--stylelosslayers",default="conv1_1;conv2_1;conv3_1;conv4_1",type=str)
    parser.add_argument("--contentlossweight",default=1.0,type=float)
    parser.add_argument("--stylelossweight",default=1.0,type=float)
    parser.add_argument("--tvlossweight",default=0,type=float)
    parser.add_argument("--usegram",default=False,type=bool)

    parser.add_argument("--imgsize",default=256,type=int)
    parser.add_argument("--learningrate",default=1e-6,type=float)
    parser.add_argument("--learningratedecay",default=5)
    parser.add_argument("--iterators",default=160000,type=int)
    parser.add_argument("--batchsize",default=10,type=int)
    parser.add_argument("--adainoutputproportion",default=1.0,type=float)

    parser.add_argument("--summaryiter",default=100,type=int)
    parser.add_argument("--saveiter",default=1000,type=int)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
