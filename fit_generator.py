import numpy as np
import pickle as pk

# ID_list = [0, 1, 2, 3, ...... 4829] total 4830 of int

class DataGenerator(object):
    def __init__(self, x, y, z, batch_size):
        self.x = x
        self.y = y
        self.z = z
        self.batch_size = batch_size

    def generate(self, ID_list, split):
        while(True):
            ind = self.shuffle(ID_list)
            ind_tr = ind[:int(ind.shape[0]*(1-split))]
            ind_val = ind[int(ind.shape[0]*(1-split)):]

            sub_batch = int(len(ind_tr)/self.batch_size)
            for i in range(sub_batch):
                ID_tmp = [ID_list[j] for j in ind_tr[i*self.batch_size:(i+1)*self.batch_size]]
                x, y, cnn = self.data_generation(ID_tmp, self.batch_size)
                x = np.concatenate([x, cnn], axis=3)

                val_ID_tmp = [ID_list[j] for j in ind_val]
                x_val, y_val, cnn_val = self.data_generation(val_ID_tmp, len(val_ID_tmp))
                x_val = np.concatenate([x_val, cnn_val], axis=3)

                yield x, y, x_val, y_val

    def data_generation(self, sub_ID, size):
        x = np.empty((size, self.x, self.y, self.z))
        cnn = np.empty((size, self.x, self.y, 2048))
        y = np.empty((size), dtype=int)

        for i, ID in enumerate(sub_ID):
            print(i, ID)
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
    x, y, cnn =a.generate(ID_list = ID_list, split=0.2)
    print('end')

