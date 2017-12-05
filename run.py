#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: run.py

import numpy as np
import argparse
import os
import cv2
import sys
from glob import glob
import csv

from tensorpack import *
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow.base import RNGDataFlow

def get_data():
    dirToView = "./csvfiles/"
    i = 0
    data = []
    for dirpath,_,filenames in os.walk(dirToView):
        for f in filenames:
            if "gazePredictions" in f:
                with open(dirToView+f, 'rb') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',')
                    print f
                    for row in spamreader:
                        # print row
                        feature = []
                        tobiiLeftEyeGazeX = float( row[2] )
                        tobiiLeftEyeGazeY = float( row[3] )
                        tobiiRightEyeGazeX = float( row[4] )
                        tobiiRightEyeGazeY = float( row[5] )
                        tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
                        tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2
                        clmTracker = row[8:len(row)-1]
                        clmTracker = [float(i) for i in clmTracker]
                        clmTrackerInt = [int(i) for i in clmTracker]
                        # Jaw
                        jaw = clmTrackerInt[0:30]
                        # for i in range(0,28,2):
                        #     cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)
                        
                        # Right eyebrow
                        reyebrow = clmTrackerInt[30:38]
                        # for i in range(30,36,2):
                        #     cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

                        # Left eyebrow
                        leyebrow = clmTrackerInt[38:46]
                        # for i in range(38,44,2):
                        #     cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

                        # Upper left eye
                        uleye = clmTrackerInt[46:52]
                        # for i in range(46,50,2):
                        #     cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

                        # Middle of left eye
                        mleye = clmTrackerInt[52:56]
                        # for i in range(54,56,2):
                        #     cv2.circle(img, (clmTrackerInt[i],clmTrackerInt[i+1]), 4, (255,0,0), -4 )        

                        # Upper right eye
                        ureye = clmTrackerInt[56:62]
                        # for i in range(56,60,2):
                        #     cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

                        # Middle of right eye
                        mreye = clmTrackerInt[62:66]
                        # for i in range(64,66,2):
                        #     cv2.circle(img, (clmTrackerInt[i],clmTrackerInt[i+1]), 4, (255,0,0), -4 ) 
                        
                        # Nose
                        nose = clmTrackerInt[66:88]
                        # for i in range(68,80,2):
                        #     cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

                        # Upper lip
                        ulip = clmTrackerInt[88:102]
                        # for i in range(88,100,2):
                        #     cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

                        # Lower lip
                        llip = clmTrackerInt[102:112]
                        # for i in range(102,110,2):
                        #     cv2.line(img, (clmTrackerInt[i],clmTrackerInt[i+1]), (clmTrackerInt[i+2],clmTrackerInt[i+3]), (0,255,0), 4)

                        feature.append(jaw)
                        feature.append(reyebrow)
                        feature.append(leyebrow)
                        feature.append(uleye)
                        feature.append(mleye)
                        feature.append(ureye)
                        feature.append(mreye)
                        feature.append(nose)
                        feature.append(ulip)
                        feature.append(llip)
                        label = [tobiiEyeGazeX, tobiiEyeGazeY]
                        data.append([feature, label])
    return data


if __name__ == '__main__':
    data = get_data()


# from vgg_model import VGGModel
# from your_model import YourModel
# import hyperparameters as hp

# """
# 15 Scene Categorization dataset declaration and loading.
# TASK: Add standardization (feature normalization)
# """
# class EyeTracker(RNGDataFlow):

#     def __init__(self, dir, name, img_size, meta_dir=None,
#                  shuffle=None, dir_structure=None):

#         assert name in ['train', 'test'], name
#         assert os.path.isdir(dir), dir
#         self.full_dir = os.path.join(dir, name)
#         self.name = name
#         assert os.path.isdir(self.full_dir), self.full_dir
#         if shuffle is None:
#             shuffle = name == 'train'
#         self.shuffle = shuffle

#         # For each category, add up to the self-enforced limit on the number of training/test examples
#         #
#         self.imglist = []
#         for catname in glob('%s/%s/*' % (dir, name)):
#             catlist = glob('%s/*' % catname)
#             c = 0
#             for fname in catlist:
#                 self.imglist.append( (fname, os.path.basename(os.path.dirname(fname))) )
#                 c = c+1
#                 if name == 'train' and c >= hp.num_train_per_category:
#                     break
#                 if name == 'test' and c >= hp.num_test_per_category:
#                     break

#         # Compact variant with no limits; just read all the data
#         # We don't do this for speed reasons
#         # self.imglist2 = [(fname, os.path.basename(os.path.dirname(fname))) for fname in glob('%s/%s/*/*' % (dir, name))]

#         self.label_lookup = dict()
#         for label in sorted(set(i[1] for i in self.imglist)):
#             self.label_lookup[label] = len(self.label_lookup)

#         self.imglist = [(fname, self.label_lookup[dirname]) for fname, dirname in self.imglist]

        
#         idxs = np.arange(len(self.imglist))

#         # Load images into numpy array
#         self.imgs = np.zeros( (img_size, img_size, 3, len(self.imglist) ), dtype=np.float )
#         for k in idxs:
#             fname, label = self.imglist[k]
#             fname = os.path.join(self.full_dir, fname)
#             img = cv2.resize( cv2.imread(fname), (img_size, img_size) )
#             img = img / 255.0 # You might want to remove this line for your standardization.
#             self.imgs[:,:,:,k] = img

#         ########################################################
#         # TASK 1: Add standardization (feature normalization).
#         mean_img = self.imgs.mean(axis=(0,1,3), keepdims=1)
#         std = self.imgs.std(axis=(0,1,3), keepdims=1)

#         for k in idxs:
#             self.imgs[:,:,:,k] -= mean_img[:,:,:,0]
#             self.imgs[:,:,:,k] /= std[:,:,:,0] 
#         ########################################################


#     def size(self):
#         return len(self.imglist)

#     def get_data(self):
#         idxs = np.arange(len(self.imglist))
#         if self.shuffle:
#             self.rng.shuffle(idxs)
#         for k in idxs:
#             fname, label = self.imglist[k]
#             fname = os.path.join(self.full_dir, fname)
#             yield [self.imgs[:,:,:,k], label]

# """
# Convenience function to load the 15 Scene database.
# This is where you would place any potential data augmentations.
# """
# def get_data(datadir, task, train_or_test):
#     isTrain = train_or_test == 'train'
#     img_size = hp.img_size
#     if task == '2':
#         img_size = 224 # Hard coded, as VGG-16 network must have this input size

#     ds = Scene15(datadir, train_or_test, img_size)
#     if isTrain:
#         augmentors = [
#             imgaug.Flip(horiz=True, prob=0.5)

#             #################################################
#             # TASK 1: Add data augmentations
#             #
#             # An example (that is duplicated work).
#             # In the Scene15 class, we resize each image to 
#             # 64x64 pixels as a preprocess. You then perform
#             # standardization over the images in Task 1.
#             #
#             # However, if we wanted to skip standardization, 
#             # we could use an augmentation to resize the image
#             # whenever it is needed:
#             # imgaug.Resize( (img_size, img_size) )
#             #
#             # Please use the same syntax to write more useful 
#             # augmentations. Read the documentation on the 
#             # TensorPack image augmentation library and experiment!
#             #################################################

#         ]
#     else:
#         # Validation/test time augmentations
#         augmentors = [
#             imgaug.Resize((img_size, img_size)) 
#         ]
#     # TensorPack: Add data augmentations
#     ds = AugmentImageComponent(ds, augmentors)
#     # TensorPack: How to batch the data
#     ds = BatchData(ds, hp.batch_size, remainder=not isTrain)
#     if isTrain:
#         # TensorPack: Perform clever image fetching, e.g., multithreaded
#         # These numbers will depend on your particular machine.
#         # Note: PrefetchData seems to be broken on Windows : /
#         if not sys.platform.lower().startswith('win'):
#             ds = PrefetchData(ds, 4, 2)
#     return ds


# """
# Program argument parsing, data setup, and training
# """
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         '--task',
#         required=True,
#         choices=['1', '2'],
#         help='Which task of the assignment to run - training from scratch (1), or fine tuning VGG-16 (2).')
#     # Set GPU to -1 to not use a GPU.
#     parser.add_argument('--gpu', help='Comma-separated list of GPU(s) to use.')
#     parser.add_argument(
#         '--load',
#         # Location of pre-trained model
#         # - As a relative path to the student distribution
#         default='vgg16.npy',
#         # - As an absolute path to the location on the Brown CS filesystem
#         # default='/course/cs1430/pretrained_weights/vgg16.npy',
#         help='Load VGG-16 model.')
#     parser.add_argument(
#         '--data',
#         # Location of 15 Scenes dataset
#         # - As a relative path to the student distribution
#         default=os.getcwd() + '/../data/',
#         # - As an absolute path to the location on the Brown CS filesystem
#         # default='/course/cs1430/datasets/15SceneData/',
#         help='Location where the dataset is stored.')

#     args = parser.parse_args()

#     if args.gpu:
#         os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

#     logger.auto_set_dir()
#     # logger.set_logger_dir('/tmp/hhe2log/train_log')

#     dataset_train = get_data(args.data, args.task, 'train')
#     dataset_test = get_data(args.data, args.task, 'test')

#     # TensorPack: Training configuration
#     config = TrainConfig(
#         model=YourModel() if args.task == '1' else VGGModel(),
#         dataflow=dataset_train,
#         callbacks=[
#             # Callbacks are performed at the end of every epoch.
#             #
#             # For instance, we can save the current model
#             ModelSaver(),
#             # Evaluate the current model and print out the loss
#             InferenceRunner(dataset_test,
#                             [ScalarStats('cost'), ClassificationError()])
#             #
#             # You can put other callbacks here to change hyperparameters,
#             # etc...
#             #
#         ],
#         max_epoch=hp.num_epochs,
#         nr_tower=max(get_nr_gpu(), 1),
#         session_config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True), allow_soft_placement=True),
#         session_init=None if args.task == '1' else get_model_loader(args.load)
#     )
#     # TensorPack: Training with simple one at a time feed into batches
#     SimpleTrainer(config).train()
