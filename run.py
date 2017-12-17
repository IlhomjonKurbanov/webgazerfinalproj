#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: run.py

import numpy as np
import argparse
import os
# import cv2
import sys
from glob import glob
import csv

from tensorpack import *
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow.base import RNGDataFlow

def get_data(train_or_test):
    #change features to dictionary?
    features = {'jaw':[], 'reyebrow':[],'leyebrow':[],'uleye':[],'mleye':[],'ureye':[],'mreye':[],'nose':[],'ulip':[],'llip':[]}
    labels = []
    if train_or_test == 'train':
        dirToView = "./csvtrain/"
    else:
        dirToView = "./csvtest/"
    i = 0
    j = 0
    for dirpath,_,filenames in os.walk(dirToView):
        for f in filenames:
            if "gazePredictions" in f:
                with open(dirToView+f, 'r') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',')
                    for row in spamreader:
                        tobiiLeftEyeGazeX = float( row[2] )
                        tobiiLeftEyeGazeY = float( row[3] )
                        tobiiRightEyeGazeX = float( row[4] )
                        tobiiRightEyeGazeY = float( row[5] )
                        tobiiEyeGazeX = (tobiiLeftEyeGazeX + tobiiRightEyeGazeX) / 2
                        tobiiEyeGazeY = (tobiiLeftEyeGazeY + tobiiRightEyeGazeY) / 2
                        clmTracker = row[8:len(row)-1]
                        clmTrackerInt = [float(i) for i in clmTracker]
                        arr = np.array(clmTrackerInt)
                        mean = np.mean(arr)
                        std = np.std(arr)
                        clmTrackerInt = [(float(i)-mean)/std for i in clmTrackerInt]
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

                        features['jaw'].append(jaw)
                        features['reyebrow'].append(reyebrow)
                        features['leyebrow'].append(leyebrow)
                        features['uleye'].append(uleye)
                        features['mleye'].append(mleye)
                        features['ureye'].append(ureye)
                        features['mreye'].append(mreye)
                        features['nose'].append(nose)
                        features['ulip'].append(ulip)
                        features['llip'].append(llip)
                        if j == 0 :
                            # print(features)
                            j += 1
                        label = [tobiiEyeGazeX, tobiiEyeGazeY]
                        # print(label)
                        labels.append(label)
    for key in features:
        features[key] = np.array(features[key])
    return (features,labels)


if __name__ == '__main__':
    data_train = get_data('train')

    # TODO is this right?
    jaw = tf.contrib.layers.real_valued_column("jaw", dimension=30, default_value=None, dtype=tf.int32, normalizer=None)
    reyebrow = tf.contrib.layers.real_valued_column("reyebrow", dimension=8, default_value=None, dtype=tf.int32, normalizer=None)
    leyebrow = tf.contrib.layers.real_valued_column("leyebrow", dimension=8, default_value=None, dtype=tf.int32, normalizer=None)
    uleye = tf.contrib.layers.real_valued_column("uleye", dimension=6, default_value=None, dtype=tf.int32, normalizer=None)
    mleye = tf.contrib.layers.real_valued_column("mleye", dimension=4, default_value=None, dtype=tf.int32, normalizer=None)
    ureye = tf.contrib.layers.real_valued_column("ureye", dimension=6, default_value=None, dtype=tf.int32, normalizer=None)
    mreye = tf.contrib.layers.real_valued_column("mreye", dimension=4, default_value=None, dtype=tf.int32, normalizer=None)
    nose = tf.contrib.layers.real_valued_column("nose", dimension=22, default_value=None, dtype=tf.int32, normalizer=None)
    ulip = tf.contrib.layers.real_valued_column("ulip", dimension=14, default_value=None, dtype=tf.int32, normalizer=None)
    llip = tf.contrib.layers.real_valued_column("llip", dimension=10, default_value=None, dtype=tf.int32, normalizer=None)

    logger.auto_set_dir()
    tf.logging.set_verbosity(tf.logging.INFO)

    # TODO is this right?
    estimator = tf.estimator.DNNRegressor(
        feature_columns=[jaw, reyebrow, leyebrow, uleye, ureye, mreye, nose, ulip, llip],
        #WHAT ARE THESE??
        hidden_units=[1024, 512, 256],
        #What is this??
        optimizer=tf.train.ProximalAdagradOptimizer(
          learning_rate=0.1,
          l1_regularization_strength=0.001
        ),
        label_dimension=2,
        model_dir = "model"
        )


    # TODO is this right??
    input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x=data_train[0],
        y=np.array(data_train[1]),
        num_epochs=100, # TODO: come back
        shuffle=True)

    # TODO what is the difference between steps and epochs
    estimator.train(input_fn=input_fn_train, steps=1000)

    data_test = get_data('test')

    input_fn_eval = tf.estimator.inputs.numpy_input_fn(
        x=data_test[0],
        y=np.array(data_test[1]),
        num_epochs=1,
        shuffle=False)

    metrics = estimator.evaluate(input_fn=input_fn_eval)

    # TODO what does the loss value here represent?
    print("Loss: %s" % metrics["loss"])

    #is there any way to understand how well our evaluations are doing step-wise?
    #how to generate log report as in proj 4?
    #what is global step?

    #parameters to check: step, num_epoch, evaluate and train params

    #seems like model is being reused at each successive training attempt?









    # print(metrics)
    # def input_fn_predict:
    # # returns x, None
    #   pass
    # predictions = estimator.predict(input_fn=input_fn_predict)

    # logger.set_logger_dir('/tmp/hhe2log/train_log')
