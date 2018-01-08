import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

import h5py
import numpy as np
import pickle as pk
from keras import regularizers
from sklearn.metrics import log_loss
from sklearn.utils import shuffle
from keras.utils import to_categorical
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten,  Reshape, TimeDistributed, Dense, Conv2D, LSTM, BatchNormalization, Dropout,\
                    Input, concatenate
from fit_generator import DataGenerator

params={
    "x": 10,
    "y": 20,
    "z": 36,
    "batch_size": 128
}

te_split = 0.1
val_split=0.1
folder = 'conf.3_20P_pkl'

def nn_model():
    #pool
    pool_in = Input(shape=(10, 74**2), name='pool_in')
    pool = Dense(2048, activation='relu')(pool_in)
    pool = Dropout(0.5)(pool)

    #conv1
    conv1_in = Input(shape=(10, 74**2))
    conv1 = Dense(2048, activation='relu')(conv1_in)
    conv1 = Dropout(0.5)(conv1)

    #conv2
    conv2_in = Input(shape=(10, 74**2))
    conv2 = Dense(2048, activation='relu')(conv2_in)
    conv2 = Dropout(0.5)(conv2)

    #dense
    dense = Input(shape=(10, 20, 2048))

    #coordinates
    coord_in = Input(shape=(10, 20, 36), name='coord_in')
    coord = concatenate([coord_in, dense], axis=3)
    coord = TimeDistributed(LSTM(100, input_shape=(20,36), dropout=0.5,\
    kernel_regularizer=regularizers.l2(0.01)), input_shape=(10,20,2084))(coord_in)#out: 3000, 20, 36
    coord = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(coord)
    coord = Dropout(0.5)(coord)

    #merge_input
    model = LSTM(100, dropout=0.5, kernel_regularizer=regularizers.l2(0.01))(coord)
    model = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)
    '''
    model = Dense(256, activation='relu')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)
    '''
    model = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)
    model = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)

    output = Dense(8, activation='softmax')(model)



    return Model(inputs=[coord_in, dense], outputs=output)

def train(batch_size=params["batch_size"], epochs=100):
    #h5f = h5py.File('/free2/p3w52016/volleyball_dataset/pkl/'+folder+'/volleyball_x.h5','r')
    #x = h5f['data'][:]
    #h5f = h5py.File('/free2/p3w52016/volleyball_dataset/pkl/'+folder+'/volleyball_y.h5','r')
    #y = h5f['label'][:]
    #print(y[:5], y.shape)
    ID_list = shuffle(np.arange(4830), random_state=0)
    ID_list = ID_list[:int(len(ID_list)*(1-te_split))]
    np.random.shuffle(ID_list)
    ID_tr = ID_list[:int(len(ID_list)*(1-val_split))]
    ID_val = ID_list[int(len(ID_list)*(1-val_split)):]

    tr_generator = DataGenerator(**params).generate(ID_list=ID_tr, mode='tr')
    val_generator = DataGenerator(**params).generate(ID_list=ID_val, mode='val')
    print('coords loaded')

    #pool = load_feature('pool', 74**2)
    #print('pool loaded')
    #conv1 = load_feature('conv1', 74**2)
    #print('cn1 loaded')
    #conv2 = load_feature('conv1', 74**2)
    #print('cn2 loaded')
    #dense = load_feature('dense', (0, 2048)
    #dense = load_feature('dense', 10**2)
    #print('den loaded')
    #print(data.shape)

    model = nn_model()
    print(model.summary())
    print('model loaded')
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.fit([x, dense], y, batch_size=batch_size, epochs=epochs, validation_split=0.05)
    model.fit_generator(generator = tr_generator,\
        steps_per_epoch = len(ID_tr)//batch_size,\
        validation_data = val_generator,\
        validation_steps = len(ID_val)//batch_size,\
        epochs=epochs)

    model.save('model.h5')
    print('model saved')

def test():
    ID_list = shuffle(np.arange(4830), random_state=0)
    ID_list = ID_list[int(len(ID_list)*(1-te_split)):]
    x=np.empty((len(ID_list), params["x"], params["y"], params["z"]))
    cnn=np.empty((len(ID_list), params["x"], params["y"], 2048))
    y=np.empty((len(ID_list)), dtype=int)
    for i, ID in enumerate(ID_list):
        print(i, ID)
        x[i] = pk.load(open('/free1/p3w52016/volleyball_dataset/pkl/conf.3_20P_pkl/%d_x.pkl' % ID, 'rb'))
        y[i] = pk.load(open('/free1/p3w52016/volleyball_dataset/pkl/conf.3_20P_pkl/%d_y.pkl' % ID, 'rb'))
        cnn[i] = pk.load(open('/free1/p3w52016/volleyball_dataset/img_feature/299_299/dense_%d.pkl' % ID, 'rb'))

    model = load_model('model.h5')
    print(model.summary)
    exit()
    pred = model.predict([x, cnn])
    loss = logloss(pred, to_categorical(y, num_classes=8))
    print(loss)


def load_feature(name, dim):
    data = np.array([]).reshape(dim)
    for i in range(55):
        tmp = pk.load(open('/free2/p3w52016/volleyball_dataset/img_feature/299_299/'+name+'_'+str(i)+'.pkl', 'rb'))
        data = np.concatenate([data, tmp])
        print(i, ' : Read!')
    return data

if __name__ == '__main__':
    #test()
    train()
    #print(nn_model().input_shape)
