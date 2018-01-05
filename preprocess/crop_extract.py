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
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, GlobalMaxPooling2D
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3

params={
    "frames_per_data": 10,
    "extract_people_num": 20
}

def load_img():
    model = xception()
    model = Model(inputs=model.input, outputs=model.output)
    #model = Model(inputs=model.input, outputs=[model.get_layer('block2_pool').output, \
    #              model.get_layer('block3_sepconv1').output, model.get_layer('block3_sepconv2').output, model.output])

    #img = cv2.imread('/free2/p3w52016/volleyball_dataset/crop/299_299/0/13456/13455_08.png', 1)#.reshape(1, 299,299,3)
    #print(img.shape)
    #exit()
    for i in range(55):#/crop/299_299/0/13286/ 20 imgs
        print('Folder: ', i)
        _dir = '/free2/p3w52016/volleyball_dataset/crop/299_299/'+str(i)+'/'
        subdir_list = sorted(os.listdir(_dir))

        #pool_o = np.array([]).reshape(0, 74*74)
        #conv1_o = np.array([]).reshape(0, 74*74)
        #conv2_o = np.array([]).reshape(0, 74*74)
        dense_o = np.array([]).reshape(0, 2048)

        for j in subdir_list:
            subdir = _dir + j #/1/ + 13286
            img_list = sorted(os.listdir(subdir))
            imgs = np.array([]).reshape(0, 299, 299, 3)

            for l in range(len(img_list)):
                img = image.load_img(subdir+'/'+img_list[l], target_size=(299, 299))
                img = image.img_to_array(img).reshape(1 ,299, 299, 3)
                print(img.shape, i, j, l)
                print(subdir+'/'+img_list[l])
                imgs = np.concatenate([imgs, img])

            dense = np.array([model.predict(imgs)])
            #pool, conv1, conv2, dense  = model.predict(imgs)
            #pool = pool.reshape(-1, 128, 74**2)
            #conv1 = conv1.reshape(-1, 256, 74**2)
            #conv2 = conv2.reshape(-1, 256, 74**2)
            dense = dense.reshape(-1, 2048, 10**2)
            #print(dense.shape)
            print('Pred:', dense.shape)

            for k in range(params["frames_per_data"]*params["extract_people_num"]):
                #pool_o = np.concatenate([pool_o, np.amax(pool[k], axis=0).reshape(1, -1)])
                #conv1_o = np.concatenate([conv1_o, np.amax(conv1[k], axis=0).reshape(1, -1)])
                #conv2_o = np.concatenate([conv2_o, np.amax(conv2[k], axis=0).reshape(1, -1)])
                dense_o = np.concatenate([dense_o, np.mean(dense[k], axis=1, dtype=float).reshape(1, -1)])
            print("Out: " ,dense_o.shape)

        '''
        with open('/free2/p3w52016/volleyball_dataset/cnn_feature/pool_%d.pkl' % i, 'wb') as f:
            pkl.dump(pool_o, f)
        with open('/free2/p3w52016/volleyball_dataset/cnn_feature/conv1_%d.pkl' % i, 'wb') as f:
            pkl.dump(conv1_o, f)
        with open('/free2/p3w52016/volleyball_dataset/cnn_feature/conv2_%d.pkl' % i, 'wb') as f:
            pkl.dump(conv2_o, f)
        '''
        with open('/free2/p3w52016/volleyball_dataset/img_feature/299_299/dense_%d.pkl' % i, 'wb') as f:
            dense_o = dense_o.reshape(-1, params["frames_per_data"], params["extract_people_num"], 2048)
            pkl.dump(dense_o, f)
        #exit()

def xception():
    return Xception(weights='imagenet', include_top=False)

def inception():
    input_tensor = Input(shape=(1280, 720, 3))
    return InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)


if __name__ == '__main__':
    load_img()
