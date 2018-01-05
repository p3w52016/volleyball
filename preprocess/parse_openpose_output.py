import numpy as np
import pickle as pkl
import pandas as pd
import json
import os
import math
import pprint

labels = {'r_set':0, 'r_spike':1, 'r-pass':2, 'r_winpoint':3, 'l_set':4, 'l-spike':5, 'l-pass':6, 'l_winpoint':7}

params = {
    "extract_people_num":20,
    "frames_per_data":10,
    "cof_th":0.4
}

def load_dir_data(dir_path='/free2/p3w52016/volleyball_dataset/data/'):
    dir_list = sorted(os.listdir(dir_path))
    for dirs in dir_list:
        dirs = dirs + '/'  #0 -> 0/
        subdir_list = sorted(os.listdir(dir_path + dirs))
        read_subdir([dir_path + dirs + subdir_list[i] for i in range(len(subdir_list))])
        print('\n\n\n')
        print(dirs + ' pass')
        print('\n\n\n')


def read_subdir(subdir_list):
    subdir_data = np.array([]).reshape(0, params["frames_per_data"], params["extract_people_num"], 36)
    annot_out = []
    #!--annot_out = np.array([labels[annot_out[i]] for i in range(len(annot_out))])

    #data/0/subdir, each sub write an X
    #--------------------------------------------------------------------------------
    for sub in subdir_list:
        #sub is full path: /free2/p3w52016/volleyball_dataset/data/0/13286
        #sub_name: 0
        sub_name = os.path.basename(os.path.dirname(sub))

        annot_in = pd.read_csv('/free2/p3w52016/volleyball_dataset/videos/'+sub_name+'/annotations.txt', \
                                sep='\n', header=None)

        file_list = sorted(os.listdir(sub))
        #target_name: 13286
        #target_index: 20 of 41
        target_name = os.path.basename(sub)
        target_index = 20
        #!--target_index = file_list.index(target_name + '_keypoints.json')

        frames = np.array([]).reshape(0, params["extract_people_num"], 36)

        #select frame and json
        #----------------------------------------------------------------------------
        for i in range(target_index - math.floor(params["frames_per_data"]/2),\
                        target_index + math.ceil(params["frames_per_data"]/2)):
            #data/0/subdir/video_folder/data_file

            with open(sub + '/' + file_list[i]) as f:
                data_file = json.load(f)

            frame_data = parse_json(data_file).reshape(-1, params["extract_people_num"], 36)
            frames = np.concatenate([frames, frame_data])

        #parse annotation for each subdir
        #----------------------------------------------------------------------------
        for i in range(len(annot_in)):
            if annot_in[0][i].split(" ")[0] == target_name + '.jpg':
                #annot_ind = i
                annot_out.append(labels[annot_in[0][i].split(" ")[1]])
                break

        #append reslut
        #----------------------------------------------------------------------------
        frames = frames.reshape(-1, params["frames_per_data"], params["extract_people_num"], 36)
        subdir_data = np.concatenate([subdir_data, frames])
        print(target_name + ' Pass')

    if len(annot_out) != subdir_data.shape[0]:
        print('Error')
        exit()

    #save as pkl from each data/subdir/, so as annotation
    with open("/free2/p3w52016/volleyball_dataset/pkl/conf.3_20P_pkl/"+ sub_name +"_x.pkl", "wb") as f:
        pkl.dump(subdir_data, f)

    with open("/free2/p3w52016/volleyball_dataset/pkl/conf.3_20P_pkl/"+ sub_name +"_y.pkl", "wb") as f:
        pkl.dump(annot_out, f)


def parse_json(json_file):
    frame_data = np.array([]).reshape(0, 36)
    mask = [i % 3 != 2 for i in range(0, 54)]

    #each detected person in single frame
    #--------------------------------------------------------------------------------
    people = json_file['people']
    for i in range(0, len(people)):
        person_kps = np.array(people[i]['pose_keypoints'])

        detected_num = int(np.count_nonzero(person_kps[mask]) / 2)
        conf_avg = sum([person_kps[j] for j in range(2, 54, 3)]) / detected_num

        #check for cof and too many undetected
        #----------------------------------------------------------------------------
        if (detected_num > 0.5 * 18) & (conf_avg > params["cof_th"]):
            frame_data = np.concatenate((frame_data, person_kps[mask].reshape(-1, 36)))

    print(frame_data.shape)
    exit()
    #slice or append to extract_people_num
    #--------------------------------------------------------------------------------
    if len(frame_data) != 0:
        mass = frame_mass(frame_data)
        frame_data = sort_frame(frame_data, mass[::2])#sort by mass x, left to right

    if len(frame_data) > params["extract_people_num"]:
        frame_data = median_filter(frame_data, mass)

    while len(frame_data) < params["extract_people_num"]:
        frame_data = np.concatenate((frame_data, np.zeros(36).reshape(1, 36)))

    return frame_data

def sort_frame(frame_data, mass):
    ind = mass.argsort()

    return frame_data[ind]


def frame_mass(frame_data):
    mass = np.array([]).reshape(0, 2*params["extract_people_num"])
    p = [frame_data[i][np.nonzero(frame_data[i])] for i in range(len(frame_data))]

    for i in range(len(p)):
        x = sum([p[i][j] for j in range(0, len(p[i]), 2)]) / ( len(p[i])/2 )
        y = sum([p[i][j] for j in range(1, len(p[i]), 2)]) / ( len(p[i])/2 )
        mass = np.append(mass, [x,y])

    return mass

def median_filter(frame_data, mass):
    filter_num = len(frame_data) - params["extract_people_num"]
    mass = mass.reshape(-1, 2)
    median = np.median(mass, axis=0)
    m_of_m = np.median(median)
    diff = np.array([mass[i]-median for i in range(len(mass))])
    diff = np.array([np.sqrt(diff[i][:-1]**2 + diff[i][-1:]**2) for i in range(len(diff))]) / m_of_m

    #Slice max diff one by one
    for i in range(filter_num):
        ind = np.argmax(diff)
        diff = np.concatenate([diff[:ind], diff[ind+1:]])
        frame_data = np.concatenate([frame_data[:ind], frame_data[ind+1:]])

    return frame_data


if __name__ == "__main__":
    load_dir_data()
