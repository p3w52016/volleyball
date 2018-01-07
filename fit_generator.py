import numpy as np
import pickle as pk
from keras.utils import to_categorical

# ID_list = [0, 1, 2, 3, ...... 4829] total 4830 of int

class DataGenerator(object):
    def __init__(self, x, y, z, batch_size):
        self.x = x
        self.y = y
        self.z = z
        self.batch_size = batch_size

    def generate(self, ID_list, mode):
        while(True):
            ind = self.shuffle(ID_list)
            sub_batch = int(len(ind)/self.batch_size) if mode == 'tr' else 1

            for i in range(sub_batch):
                if mode == 'tr':
                    ID_tmp = [ID_list[j] for j in ind[i*self.batch_size:(i+1)*self.batch_size]]
                    x, y, cnn = self.data_generation(ID_tmp, self.batch_size)
                elif mode =='val':
                    ID_tmp = [ID_list[j] for j in ind]
                    x, y, cnn = self.data_generation(ID_tmp, len(ind))

                yield [x, cnn], to_categorical(y, num_classes=8)

    def data_generation(self, sub_ID, size):
        x = np.empty((size, self.x, self.y, self.z))
        cnn = np.empty((size, self.x, self.y, 2048))
        y = np.empty((size), dtype=int)

        for i, ID in enumerate(sub_ID):
            x[i] = pk.load(open('/free1/p3w52016/volleyball_dataset/pkl/conf.3_20P_pkl/%d_x.pkl' % i, 'rb'))
            y[i] = pk.load(open('/free1/p3w52016/volleyball_dataset/pkl/conf.3_20P_pkl/%d_y.pkl' % i, 'rb'))
            cnn[i] = pk.load(open('/free1/p3w52016/volleyball_dataset/img_feature/299_299/dense_%d.pkl' % i, 'rb'))

        return x, y, cnn

    def shuffle(self, ID_list):
        ind = np.arange(len(ID_list))
        np.random.shuffle(ind)

        return ind

if __name__ == "__main__":
    a = DataGenerator(x=10, y=20, z=36, batch_size=512)
    ID_list = [i for i in range(4830)]
    x, y=a.generate(ID_list = ID_list[int(4830*0.9):], mode='val')
    print(x[0].shape, x[1].shape, y.shape)

