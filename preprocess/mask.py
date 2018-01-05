import numpy as np
import pickle as pk
import cv2
import os

params={
    "extract_people_num": 20,
    "frames_per_data": 10,
    "box_w": 299,
    "box_h": 299
}

def mask_bkg():
    for i in range(55):#dir_name: /0/, /1/
        path = '/free2/p3w52016/volleyball_dataset/pkl/conf.3_20P_pkl/%d_x.pkl' % i
        subdir_list = sorted(os.listdir('/free2/p3w52016/volleyball_dataset/bkg/%d/' % i))
        print(i)
        with open(path, 'rb') as f:
            x = pk.load(f)[:, :, :, :]
        ct = mass(x)
        #print(ct.shape, ct[2][0])
        for j in range(x.shape[0]):#subdir_name: /13286
            img_list = sorted(os.listdir('/free2/p3w52016/volleyball_dataset/videos/'+str(i)+'/'+subdir_list[j]))
            print(subdir_list[j])
            for k in range(20-int(params["frames_per_data"]/2), 20+int(params["frames_per_data"]/2), 1):
            #15~25 index, total 10 frames
                img = cv2.imread('/free2/p3w52016/volleyball_dataset/videos/'+str(i)+'/'+subdir_list[j]+'/'+\
                                    img_list[k], 1)
                pts = ct[j][k-15]
                print(img_list[k])
                for l in range(params["extract_people_num"]):#20 people in a frame
                    x1 = int(pts[l][0]-params["box_w"]/2) if np.any(pts[l][0] >= params["box_w"]/2) else 0
                    y1 = int(pts[l][1]-params["box_h"]/2) if np.any(pts[l][1] >= params["box_h"]/2) else 0
                    x_b = x1+params["box_w"]
                    y_b = y1+params["box_h"]
                    #print(x1, x_b, y1, y_b)

                    if (pts[l][0]==0)&(pts[l][1]==0):
                        crop = np.zeros((params["box_h"], params["box_w"], 3))
                    elif(x1+params["box_w"]<img.shape[1])&(y1+params["box_h"]<img.shape[0]):
                        crop = img[y1:y_b, x1:x_b]
                    elif(x1+params["box_w"]>img.shape[1]):
                        crop = img[y1:y_b, img.shape[1]-params["box_w"]:img.shape[1]]
                    elif(y1+params["box_h"]>img.shape[0]):
                        crop = img[img.shape[0]-params["box_h"]:img.shape[0], x1:x_b]
                    else:
                        crop = img[img.shape[0]-params["box_h"]:img.shape[0], img.shape[1]-params["box_w"]:img.shape[1]]

                    if l < 10:
                        cv2.imwrite('/free2/p3w52016/volleyball_dataset/crop/299_299/'+str(i)+'/'+subdir_list[j]+'/'+\
                                    img_list[k][:-4]+'_0%d.png' % l, crop)
                    else:
                        cv2.imwrite('/free2/p3w52016/volleyball_dataset/crop/299_299/'+str(i)+'/'+subdir_list[j]+'/'+\
                                    img_list[k][:-4]+'_%d.png' % l, crop)

                #exit()

def mass(coord):
    people_num = coord.shape[0]*coord.shape[1]*coord.shape[2]
    mass = np.array([], dtype=np.float64)
    coord = coord.reshape(-1)
    for i in range(0, people_num):
        kp_num = np.count_nonzero(coord[i*36:(i+1)*36]) / 2
        if kp_num == 0: kp_num = 1
        x = sum([coord[j] for j in range(i*36, (i+1)*36, 2)]) / kp_num
        y = sum([coord[j] for j in range((i*36)+1, (i+1)*36+1, 2)]) / kp_num
        mass = np.append(mass, x)
        mass = np.append(mass, y)

    return  mass.reshape(-1, params["frames_per_data"], params["extract_people_num"], 2)

if __name__ == '__main__':
    mask_bkg()
