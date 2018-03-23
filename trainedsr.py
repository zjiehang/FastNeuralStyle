# coding : utf-8
from __future__ import print_function
from __future__ import division
import sys
import tensorflow as tf
import argparse
import os
from utils.data import Data
from model.edsrmodel import EDSRModel

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
FLAGS = None

def main(_):
    data = Data(FLAGS.contentpath,FLAGS.stylepath,FLAGS.imgsize)
    network = EDSRModel(FLAGS.pretrainedpath,
                         FLAGS.adainoutputproportion,
                         FLAGS.edsrlayer,
                         FLAGS.edsrfeaturesize,
                         FLAGS.contentlosslayer,
                         FLAGS.stylelosslayers,
                         FLAGS.contentlossweight,
                         FLAGS.stylelossweight,
                         FLAGS.tvlossweight,
                         FLAGS.usegram,
                         FLAGS.batchsize,
                         FLAGS.learningrate,
                         FLAGS.learningratedecay)
    network.buildTrainModel()
    network.set_data(data)
    network.train(FLAGS.batchsize,
                  FLAGS.iterators,
                  FLAGS.savedir,
                  False,
                  FLAGS.reusedir,
                  FLAGS.logdir,
                  FLAGS.summaryiter,
                  FLAGS.saveiter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--edsrlayer",default=16,type=int)
    parser.add_argument("--edsrfeaturesize", default=128, type=int)

    parser.add_argument("--contentpath",default="images/content",type=str)
    parser.add_argument("--stylepath",default="images/style",type=str)
    parser.add_argument("--pretrainedpath",default="pretrained/vgg19_weights_normalized.h5",type=str)
    parser.add_argument("--contentlosslayer",default="conv4_1",type=str)
    parser.add_argument("--stylelosslayers",default="conv1_1;conv2_1;conv3_1;conv4_1",type=str)
    parser.add_argument("--contentlossweight",default=10.0,type=float)
    parser.add_argument("--stylelossweight",default=1e-2,type=float)
    parser.add_argument("--tvlossweight",default=0.0,type=float)
    parser.add_argument("--usegram",default=False,type=bool)

    parser.add_argument("--imgsize",default=256,type=int)
    parser.add_argument("--learningrate",default=1e-4,type=float)
    parser.add_argument("--learningratedecay",default=5e-5,type=float)
    parser.add_argument("--iterators",default=10000,type=int)
    parser.add_argument("--batchsize",default=10,type=int)
    parser.add_argument("--adainoutputproportion",default=1.0,type=float)

    parser.add_argument("--summaryiter",default=100,type=int)
    parser.add_argument("--saveiter",default=1000,type=int)

    parser.add_argument("--savedir",default='ckpt/adain_test',type=str)
    parser.add_argument("--logdir",default='logfile/adain_10000_test',type=str)
    parser.add_argument("--reusedir", default='ckpt/adain_10000_test', type=str)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
