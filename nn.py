import h5py
import numpy as np
import pickle as pk
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Flatten,  Reshape, TimeDistributed, Dense, Conv2D, LSTM, BatchNormalization, Dropout,\
                    Input, concatenate
folder = 'conf.4_20P_pkl'

def nn_model():
    #coordinates
    coord_in = Input(shape=(10, 20, 36), name='coord_in')
    coord = TimeDistributed(LSTM(1000, input_shape=(20,36), dropout=0.5), input_shape=(10,20,36))(coord_in)#out: 3000, 20, 36
    coord = Dense(512, activation='relu')(coord)
    coord = Dropout(0.5)(coord)

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
    dense = Input(shape=(10, 2048))

    #merge_input
    model = concatenate([dense, coord])

    model = LSTM(3000, dropout=0.5)(model)
    model = Dense(512, activation='relu')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)
    model = Dense(256, activation='relu')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)
    model = Dense(128, activation='relu')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)
    model = Dense(64, activation='relu')(model)
    model = BatchNormalization()(model)
    model = Dropout(0.5)(model)

    output = Dense(8, activation='softmax')(model)



    return Model(inputs=[coord_in, dense], outputs=output)

def train(batch_size=512, epochs=1000):
    #h5f = h5py.File('/free2/p3w52016/volleyball_dataset/pkl/'+folder+'/volleyball_x.h5','r')
    #x = h5f['data'][:]
    h5f = h5py.File('/free2/p3w52016/volleyball_dataset/pkl/'+folder+'/volleyball_y.h5','r')
    y = h5f['label'][:]
    print(y[:5], y.shape)
    exit()
    y = to_categorical(y, num_classes=8)
    print('coords loaded')

    #pool = load_feature('pool', 74**2)
    print('pool loaded')
    #conv1 = load_feature('conv1', 74**2)
    print('cn1 loaded')
    #conv2 = load_feature('conv1', 74**2)
    print('cn2 loaded')
    #dense = load_feature('dense', (0, 2048)
    dense = load_feature('dense', (0, 10, 20, 2048))
    print('den loaded')
    print(data.shape)
    exit()

    model = nn_model()
    print(model.summary())
    print('model loaded')
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit([x, dense], y, batch_size=batch_size, epochs=epochs, validation_split=0.05)

def load_feature(name, dim):
    data = np.array([]).reshape(dim)
    for i in range(55):
        tmp = pk.load(open('/free2/p3w52016/volleyball_dataset/img_feature/299_299/'+name+'_'+str(i)+'.pkl', 'rb'))
        data = np.concatenate([data, tmp])
        print(i, ' : Read!')
    return data

if __name__ == '__main__':
    train()
    #nn_model()
