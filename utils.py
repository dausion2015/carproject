from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pdb
import glob

# import caffe
import scipy.misc
import numpy as np
import tensorflow as tf
import argparse
import inception_preprocessing
import math
parser = argparse.ArgumentParser()
args, unparsed = parser.parse_known_args()



# def extract(model, weights):
#     """extract from .caffemodel base on .prototxt to .npy, don't extract original vgg's fc layers
#     Args:
#         model: the .prototxt file path
#         weights: the .caffemodel file path
#     """
#     net = caffe.Net(model, 1, weights=weights)

#     parameters = {}
#     for layer, param in net.params.iteritems():
#         if 'fc6' == layer or 'fc7' == layer or 'fc8' == layer:
#             continue
#         w = param[0].data
#         b = param[1].data
#         parameters[layer] = [w, b]

#     if 'CAM' in model:
#         filename = 'vgg16CAM_train_iter_90000.npy'
#     else:
#         filename = 'VGG_ILSVRC_16_layers.npy'
#     np.save(os.path.join(args.pretrain, filename), parameters)
#     # np.save(os.path.join(args.pretrain, filename), parameters)
#     return 'model .npy file generate sucessfull on'+os.path.join(args.pretrain, filename)+'!!!!!'


class data_loader():
    def __init__(self, flag, batch_size,  dataset_dir, num_epochs,num_threads=4):
    
        self.flag = flag
        # self.num = num

        if self.flag:
            filename = [os.path.join(dataset_dir,'quiz_train_0000{}of00004.tfrecord'.format(str(i))) for i in range(4)] #train dataset
            self.num_batches = math.floor(43971 / batch_size)
        else:
            filename = [os.path.join(dataset_dir,'quiz_validation_0000{}of00004.tfrecord'.format(str(i))) for i in range(4)] #validation dataset
            self.num_batches = math.floor(4885 / batch_size)
            # self.images, self.labels = self.readFromTFRecords(self.flag,os.path.join(args.dataset_dir, filename),
                # batch_size, num_epochs, [224, 224, 3], num_threads)
        self.images, self.labels = self.readFromTFRecords(self.flag,filename,num_epochs, [224, 224, 3],
                                                         batch_size, num_threads)

    

    #  def readFromTFRecords(self,self.flag, filename, batch_size, num_epochs, img_shape, num_threads, min_after_dequeue=10000):
    def readFromTFRecords(self, flag, filename,  num_epochs,  img_shape, batch_size, num_threads,min_after_dequeue=100):
        if flag:
            num_epochs = num_epochs
        else:
            num_epochs = 1
        filename_queue = tf.train.string_input_producer(filename, num_epochs=num_epochs)

        # def read_and_decode(filename_queue, img_shape):
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/class/label': tf.FixedLenFeature([], tf.int64),
                    'image/encoded': tf.FixedLenFeature([], tf.string)
                }
        )
        # image = tf.image.decode_jpeg(features['image/encoded'],3)
        # image = tf.reshape(image, img_shape)    # THIS IS IMPORTANT
        # image = tf.cast(image, tf.float32) 
        
        image = tf.image.decode_jpeg(features['image/encoded'], 3)
        
        image = inception_preprocessing.preprocess_image(image,img_shape[0], img_shape[1], is_training=self.flag,)
       
        sparse_label = features['image/class/label']       # tf.int64
            # return image, sparse_label

        # image, sparse_label = read_and_decode(filename_queue, img_shape) # share filename_queue with multiple threads

        # tf.train.shuffle_batch internally uses a RandomShuffleQueue
        images, sparse_labels = tf.train.shuffle_batch(
                [image, sparse_label], batch_size=batch_size, num_threads=num_threads,
                min_after_dequeue=min_after_dequeue,
                capacity=min_after_dequeue + (num_threads+1)*batch_size
        )
        
        return images, sparse_labels    

