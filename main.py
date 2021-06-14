# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import glob
from imageio import imread, imsave
import cv2
import argparse
import dlib
import sys
import bz2
from tensorflow.keras.utils import get_file
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--no_makeup', type=str, default=os.path.join('imgs', 'no_makeup', 'xfsy_0068.png'), help='path to the no_makeup image')
args = parser.parse_args()

def preprocess(img):
    return (img / 255. - 0.5) * 2

def deprocess(img):
    return (img + 1) / 2

batch_size = 1
img_size = 256

#face_alignment
LANDMARKS_DATA_FILENAME = "shape_predictor_68_face_landmarks.dat"
LANDMARKS_MODEL_URL = f'http://dlib.net/files/{LANDMARKS_DATA_FILENAME}.bz2'
if not Path(LANDMARKS_DATA_FILENAME).exists():
    data_bz = get_file(f'{LANDMARKS_DATA_FILENAME}.bz2', 
                                               LANDMARKS_MODEL_URL, cache_subdir='temp')
    with open(LANDMARKS_DATA_FILENAME, 'bw') as f: 
        f.write(bz2.open(data_bz).read())
landmarks_detector = LandmarksDetector(LANDMARKS_DATA_FILENAME)

from PIL import Image
from IPython.display import display

path = args.no_makeup
for face_landmarks in landmarks_detector.get_landmarks(path):
    aligned_img = image_align(path, face_landmarks)
    aligned_img = aligned_img.resize((256,256))
    # display(aligned_img, Image.LANCZOS)

no_makeup = np.array(aligned_img).astype("uint8")
#no_makeup = cv2.resize(imread(args.no_makeup), (img_size, img_size))
X_img = np.expand_dims(preprocess(no_makeup), 0)
makeups = glob.glob(os.path.join('imgs', 'makeup', '*.*'))
result = np.ones((2 * img_size, (len(makeups) + 1) * img_size, 3))
result[img_size: 2 *  img_size, :img_size] = no_makeup / 255.

tf.reset_default_graph()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph(os.path.join('model', 'model.meta'))
saver.restore(sess, tf.train.latest_checkpoint('model'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')

for i in range(len(makeups)):
    makeup = cv2.resize(imread(makeups[i]), (img_size, img_size))
    Y_img = np.expand_dims(preprocess(makeup), 0)
    Xs_ = sess.run(Xs, feed_dict={X: X_img, Y: Y_img})
    Xs_ = deprocess(Xs_)
    result[:img_size, (i + 1) * img_size: (i + 2) * img_size] = makeup / 255.
    result[img_size: 2 * img_size, (i + 1) * img_size: (i + 2) * img_size] = Xs_[0]

imsave(path.split("\\")[-1].split(".")[0]+ "_beautygan.png", result)
