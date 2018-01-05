import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

import cv2
import numpy as np
import os
import math
import h5py
import pickle as pkl
from keras.models import Model
from keras.layers import Input, GlobalMaxPooling2D
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

params={
    "frames_per_data": 10
}

def load_img():
    model = xception()
    model = Model(inputs=model.input, outputs=[model.get_layer('block2_pool').output, \
                    model.get_layer('block3_sepconv1').output, model.get_layer('block3_sepconv2').output, model.output])
    #model = inception()


    for i in range(55):
        print('Folder: ', i)
        _dir = '/free2/p3w52016/volleyball_dataset/crop/299_299/' + str(i)
        subdir_list = open(_dir + '/list')

        #pool_o = np.array([]).reshape(0, 74*74)
        #conv1_o = np.array([]).reshape(0, 74*74)
        #conv2_o = np.array([]).reshape(0, 74*74)
        dense_o = np.array([]).reshape(0, 2048)

        for j in subdir_list:
            j = j.rstrip()
            if (j!='list') & (j!='annotations.txt'):
                print('Sub: ', j)
                subdir = _dir + '/' + j
                img_list = sorted(os.listdir(subdir))
                target_ind = 20

                imgs = np.array([]).reshape(0, 299, 299, 3)
                for k in range(target_ind - math.floor(params["frames_per_data"]/2),\
                                target_ind + math.ceil(params["frames_per_data"]/2)):
                    img = subdir + '/' + img_list[k]
                    img = cv2.imread(img, 1)
                    img = cv2.resize(img, (299, 299), interpolation= cv2.INTER_CUBIC).reshape(1, 299, 299, 3)
                    imgs = np.concatenate([imgs, img])

                pool, conv1, conv2, dense  = model.predict(imgs)
                pool = pool.reshape(-1, 128, 74**2)
                conv1 = conv1.reshape(-1, 256, 74**2)
                conv2 = conv2.reshape(-1, 256, 74**2)
                dense = dense.reshape(-1, 2048, 10**2)

                for k in range(params["frames_per_data"]):
                    pool_o = np.concatenate([pool_o, np.amax(pool[k], axis=0).reshape(1, -1)])
                    conv1_o = np.concatenate([conv1_o, np.amax(conv1[k], axis=0).reshape(1, -1)])
                    conv2_o = np.concatenate([conv2_o, np.amax(conv2[k], axis=0).reshape(1, -1)])
                    dense_o = np.concatenate([dense_o, np.mean(dense[k], axis=1, dtype=float).reshape(1, -1)])
                #print(pool_o.shape, conv1_o.shape, conv2_o.shape, dense_o.shape)
        '''
        with open('/free2/p3w52016/volleyball_dataset/cnn_feature/pool_%d.pkl' % i, 'wb') as f:
            pkl.dump(pool_o, f)
        with open('/free2/p3w52016/volleyball_dataset/cnn_feature/conv1_%d.pkl' % i, 'wb') as f:
            pkl.dump(conv1_o, f)
        with open('/free2/p3w52016/volleyball_dataset/cnn_feature/conv2_%d.pkl' % i, 'wb') as f:
            pkl.dump(conv2_o, f)
        '''
        with open('/free2/p3w52016/volleyball_dataset/img_feature/299_299/dense_%d.pkl' % i, 'wb') as f:
            pkl.dump(dense_o, f)

def xception():
    return Xception(weights='imagenet', include_top=False)

def inception():
    input_tensor = Input(shape=(1280, 720, 3))
    return InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)


if __name__ == '__main__':
    load_img()
