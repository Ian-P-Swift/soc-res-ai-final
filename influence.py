import sys
import time
from threading import Thread

import tensorflow as tf
import numpy as np
import keras
from tensorflow.python.eager import forwardprop
import dill
import data

SKIN_TONE_IMAGE = "TRAIN0403.png"
GENDER_IMAGE = "TRAIN0359.png"
AGE_IMAGE = "TRAIN0300.png"
USE_PIPELINE_1 = False

def hvp1(model, images, labels, vector):
    with forwardprop.ForwardAccumulator(
            model.trainable_variables, vector) as acc:
        with tf.GradientTape() as grad_tape:
            pred = model(images, training=True)
            loss = tf.losses.categorical_crossentropy(
                labels, pred)
        grads = grad_tape.gradient(loss, model.trainable_variables)
    return acc.jvp(grads)

def vector(X, Y, model):
    with tf.GradientTape() as t1:
        pred = model(X)
        loss = tf.losses.categorical_crossentropy(Y, pred)
    return t1.gradient(loss, model.trainable_weights)

def calculate_uploss(X, Y, model, sample, index):
    v_test = vector(np.array([X[index, :]]), np.array([Y[index, :]]), model)
    v_reshaped = [tf.reshape(v_test[i], [-1]) for i in range(len(v_test))]
    v_test = tf.concat(v_reshaped, axis=0)

    v = vector(np.array([X[sample, :]]), np.array([Y[sample, :]]), model)
    v_prev = v
    np.random.seed(42)

    nn = X.shape[0]
    for i in range(3):
        nextSample = np.random.randint(nn - 1)
        hvp_vprev = hvp1(model, np.array([X[nextSample, :]]), np.array([Y[nextSample, :]]), v_prev)
        v_next = [tf.add(tf.add(v[i], hvp_vprev[i]), v_prev[i]) for i in range(len(v))]
        v_prev = v_next

    v_prev_reshaped = [tf.reshape(v_prev[i], [-1]) for i in range(len(v_prev))]
    v_prev = tf.concat(v_prev_reshaped, axis=0)

    return -1*tf.tensordot(v_test, v_prev, axes=1)


def get_influence(model_name, file_name, cat_id, outfile):
    d = data.Data()
    d.load_images('./train/')
    X, Y = d.X, d.Y
    if USE_PIPELINE_1:
        d.load_faces()
        X, Y = d.get_faces()
    index = d.get_index_from_filename(file_name, X)

    model = keras.models.load_model(model_name)
    scores = [None]*X.shape[0]
    for i in range(X.shape[0]):
        scores[i] = calculate_uploss(X, Y[cat_id], model, i, index)
        if i%100 == 0:
            print('.')
    dill.dump(scores, open(outfile, 'wb'))


threads = [None] * 3
threads[0] = Thread(target=get_influence, args=(('./models/model_skin_tone.h5' if USE_PIPELINE_1 else './models/model_skin_tone__smoteless_35.h5'), './train/'+SKIN_TONE_IMAGE, 0, 'uploss_scores_skin_tone.dill'))
threads[1] = Thread(target=get_influence, args=(('./models/model_gender.h5' if USE_PIPELINE_1 else './models/model_gender__smoteless_35.h5'), './train/'+GENDER_IMAGE, 1, 'uploss_scores_gender.dill'))
threads[2] = Thread(target=get_influence, args=(('./models/model_age.h5' if USE_PIPELINE_1 else './models/model_age__smoteless_35.h5'), './train/'+AGE_IMAGE, 2, 'uploss_scores_age.dill'))

threads[0].start()
threads[1].start()
threads[2].start()

for i in range(3):
    threads[i].join()

print('done!')