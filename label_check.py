import pandas as pd
from random import choice,sample,shuffle
import numpy as np
import json
import random
import math
import os
import re

def triplet_extraction(class_label):
        exist_labels = np.zeros(class_label.shape[-1]) - 1
        position_list = []
        for i in range(class_label.shape[1]):
            temp_list = []
            ### extract the exist label for each entity and maintain -1 if not mentioned. ###
            if 0 in class_label[:,i]:
                exist_labels[i] = 0
                
            if 1 in class_label[:,i]:
                exist_labels[i] = 1
                ### if the entity exists try to get its position.### 
                ### Note that, the contrastive loss will only be caculated on exist entity as it is meaningless to predict their position for the non-exist entities###
                temp_list.append(random.choice(np.where(class_label[:,i] == 1)[0]))
                try:
                    temp_list = temp_list + random.sample(np.where(class_label != 1)[0].tolist(),7)
                except:
                    print('fatal error')
            if temp_list == []:
                temp_list = temp_list +random.sample(np.where(class_label != 1)[0].tolist(),8)
            position_list.append(temp_list)
        return exist_labels, position_list


label_path = '/remote-home/mengxichen/UniBrain-lora/Pretrain/data_file_more_label/label.npy'
# label_all = np.load(label_path,allow_pickle=True)
# print(label_all[0,:,:])
# print(triplet_extraction(label_all[0,:,:])[0])
# report_path = '/remote-home/mengxichen/UniBrain-lora/Pretrain/data_file_more_label/report_observe_fuse_global.npy'
# report = np.load(report_path,allow_pickle=True)
# print(report.item())
# print(type(report))
a = np.load('missidx_t.npy')
print(a[:10])