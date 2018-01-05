import numpy as np

class DataGenerator(object):
    def __init__(self, x, y, z, batch_size):
        self.x = x
        self.y = y
        self.z = z
        self.batch_size = batch_size

    def genrate(self, labels, ID_list):
        while(True):
            ind = self.shffle(ID_list)

            loop = int(len(ind)/self.batch_size)
            for i in range(loop):
                sub_list = [ID_list[j] for j in ind[i*self.batch_size:(i+1)*self.batch_size]]

                x, y =


    def shuffle(self, ID_list):
        ind = np.arrange(len(ID_list))
        np.random.shffle(ind)

        return ind
    def
