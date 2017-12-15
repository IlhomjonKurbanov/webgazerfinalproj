from tensorpack import *
from tensorpack.tfutils import get_model_loader
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.dataflow.base import RNGDataFlow
import tensorflow as tf
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)
model = None



@app.route('/')
def index():
    load_estimator()

    return render_template('demo.html')

@app.route('/_get_predictions')
def get_prediction():
    global estimator

    ### TODO: TAILOR DATA TO AJAX

    # TODO: Convert JSON AJAX input to python dict:

    # Convert input data to nparray dict format
    for key in inputdata:
        inputdata[key] = np.array(inputdata[key])

    input_fn_eval = tf.estimator.inputs.numpy_input_fn(
        x=testdata,
        num_epochs=1,
        shuffle=False)

    pred = estimator.predict(input_fn_eval)

    # TODO: SEND AJAX REQUEST WITH ESTIMATOR DATA

    # for i, p in enumerate(pred):
    # 	print("Prediction %s" % (p))

    ####


def load_estimator():
    global estimator

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
        model_dir = "model")


    testdata = {'mreye': [[429, 116, 426, 104]],
     'ulip': [[334, 253, 351, 241, 372, 237, 388, 237, 403, 234, 425, 234, 443, 243]],
     'leyebrow': [[265, 104, 287, 99, 316, 104, 339, 109]],
     'uleye': [[283, 118, 306, 101, 333, 120]],
     'llip': [[436, 261, 420, 274, 395, 279, 369, 277, 349, 269]],
     'jaw': [[241, 118, 249, 167, 260, 212, 276, 256, 299, 292, 327, 320, 359, 344, 404, 350, 443, 334, 460, 306, 472, 274, 481, 237, 484, 193, 485, 148, 482, 100]],
     'mleye': [[306, 125, 306, 112]],
     'nose': [[369, 116, 350, 172, 340, 194, 351, 206, 381, 211, 408, 200, 416, 186, 404, 168, 375, 156, 357, 199, 404, 195]],
     'ureye': [[452, 106, 426, 93, 402, 115]],
     'reyebrow': [[464, 91, 443, 90, 416, 98, 395, 106]]}


    #sess = tf.Session()

    # saver = tf.train.import_meta_graph('model/model.ckpt-62000.meta')
    # saver.restore(sess, tf.train.latest_checkpoint('model/'))
    #
    # graph = tf.get_default_graph()
    #
    #
    # testdata = {'mreye': [[429, 116, 426, 104]],
    #  'ulip': [[334, 253, 351, 241, 372, 237, 388, 237, 403, 234, 425, 234, 443, 243]],
    #  'leyebrow': [[265, 104, 287, 99, 316, 104, 339, 109]],
    #  'uleye': [[283, 118, 306, 101, 333, 120]],
    #  'llip': [[436, 261, 420, 274, 395, 279, 369, 277, 349, 269]],
    #  'jaw': [[241, 118, 249, 167, 260, 212, 276, 256, 299, 292, 327, 320, 359, 344, 404, 350, 443, 334, 460, 306, 472, 274, 481, 237, 484, 193, 485, 148, 482, 100]],
    #  'mleye': [[306, 125, 306, 112]],
    #  'nose': [[369, 116, 350, 172, 340, 194, 351, 206, 381, 211, 408, 200, 416, 186, 404, 168, 375, 156, 357, 199, 404, 195]],
    #  'ureye': [[452, 106, 426, 93, 402, 115]],
    #  'reyebrow': [[464, 91, 443, 90, 416, 98, 395, 106]]}
    #
    #
    # input_fn_eval = tf.estimator.inputs.numpy_input_fn(
    #     x=testdata,
    #     y=None,
    #     num_epochs=1,
    #     shuffle=False)
    #
    # #metrics = estimator.evaluate(input_fn=input_fn_eval)
    #
    # print graph.predict(input_fn_eval)
