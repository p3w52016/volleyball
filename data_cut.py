import numpy as np
import pickle as pk




def cut(mode):
    count=0
    if mode == 'locat':
        for i in range(55):
            x=np.array([pk.load(open('/free2/p3w52016/volleyball_dataset/pkl/conf.3_20P_pkl/%d_x.pkl' %i, 'rb'))]).reshape(-1, 10, 20, 36)
            y=np.array([pk.load(open('/free2/p3w52016/volleyball_dataset/pkl/conf.3_20P_pkl/%d_y.pkl' %i, 'rb'))]).reshape(-1, 1)

            for j in range(len(x)):
                pk.dump(x[j:j+1, :, :, :], \
                    open('/free1/p3w52016/volleyball_dataset/pkl/conf.3_20P_pkl/%d_x.pkl' % count, 'wb'))
                pk.dump(y[j:j+1, :], \
                    open('/free1/p3w52016/volleyball_dataset/pkl/conf.3_20P_pkl/%d_y.pkl' % count, 'wb'))
                count = count+1
                print(i, j, count)

    elif mode == 'cnn':
        for i in range(55):
            x=np.array([pk.load(open('/free2/p3w52016/volleyball_dataset/img_feature/299_299/dense_%d.pkl' %i, 'rb'))]).reshape(-1, 10, 20, 2048)

            for j in range(len(x)):
                pk.dump(x[j:j+1, :, :, :], \
                    open('/free1/p3w52016/volleyball_dataset/img_feature/299_299/dense_%d.pkl' % count, 'wb'))
                count = count+1
                print(i, j, count)

if __name__ == '__main__':
    cut(mode='cnn')

