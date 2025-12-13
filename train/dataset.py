import os
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import sys
import random
import glob, os
import augmentation
import unicodedata

# Get the path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get the path of the parent directory
parent_dir = os.path.dirname(script_dir)

# Add the parent directory to the Python path
sys.path.append(parent_dir)

from utils.data_processing import Preprocess_Module


class DataSet_Classification(Dataset):
    def __init__(self, data_list, root_dir, transform=None, data_augmentation = False):
        self.npy_data = []
        self.annotations = []
        self.data_list = data_list
        self.root_dir = root_dir
        for video_data in self.data_list:
            video_path = os.path.join(self.root_dir, video_data.split('+++')[0])
            video_anno = video_data.split('+++')[1].strip()
            npy_ls = sorted(glob.glob(os.path.join(video_path, "*.npy")))
            self.npy_data.extend(npy_ls)
            self.annotations.extend([video_anno] * len(npy_ls))
        self.transform = Preprocess_Module(data_augmentation)
        
    def __len__(self):
        
        # return 30
        
        return len(self.npy_data)
        # return 10
    
    def __getitem__(self, idx):
        # temp_line = self.npy_data[idx].strip()
        # annotations = temp_line.split('+++')
        # data_file_path = os.path.join('../data/'+self.root_dir, annotations[0].strip('.'))
        #data_file_path = data_file_path.replace("new_takeda_processed_coco_merged", "new_takeda_mediapipe_processed_coco_merged")
        # label_gmfcs = int(annotations[1].rstrip()) - 1
        # label_moca = float(annotations[2])
        # label_best = float(annotations[3])
        # label_best_gait = float(annotations[4])
        data_file_path = self.npy_data[idx]
        label_gmfcs = int(self.annotations[idx]) - 1
        #path_nfc = unicodedata.normalize('NFC', data_file_path)
        data = np.load(data_file_path)
        tmp_dict = {}
        tmp_dict['img_shape'] = (800, 1422)
        tmp_dict['label'] = -1
        tmp_dict['start_index'] = 0
        tmp_dict['modality'] = 'Pose'
        tmp_dict['total_frames'] = 124
        # data = np.where(data == 0, 1e-4, data)
        data[np.isnan(data)] = 0
        # print(data)
        tmp_dict['keypoint'] = data[np.newaxis, :, :, :2]
        tmp_dict['keypoint'] = np.tile(tmp_dict['keypoint'], (2, 1, 1, 1))
        # print(tmp_dict['keypoint'].shape)
        tmp_dict['keypoint_score'] = data[np.newaxis, :, :, 2] #because we do not have class 0
        tmp_dict['keypoint_score'] = np.tile(tmp_dict['keypoint_score'], (2, 1, 1))
        # print(tmp_dict['keypoint_score'].shape)
        
        data = self.transform(tmp_dict)
        data = data['keypoint'][0]
        # y_onehot = torch.nn.functional.one_hot(torch.tensor(label), num_classes=5)
        # print(type(label))

        return data, label_gmfcs


class DataSet_Classification_upper(Dataset):
    def __init__(self, npy_file_paths, root_dir, transform=None, data_augmentation=False):
        # self.npy_data = np.load(npy_file_path)
        self.npy_data = np.concatenate([np.load(path) for path in npy_file_paths])
        self.root_dir = root_dir
        self.transform = Preprocess_Module(data_augmentation)

    def __len__(self):
        # return 30
        return len(self.npy_data)
        # return 10

    def __getitem__(self, idx):
        label, _, video, clip = self.npy_data[idx]
        label -= 1
        data_file_path = os.path.join(self.root_dir, str(video), str(video) + '_' + str(clip) + '.npy')
        # print(data_file_path)
        data = np.load(data_file_path)
        # print(data_file_path)
        # print(data)
        tmp_dict = {}
        tmp_dict['img_shape'] = (320, 480)
        tmp_dict['label'] = -1
        tmp_dict['start_index'] = 0
        tmp_dict['modality'] = 'Pose'
        tmp_dict['total_frames'] = 124
        # data = np.where(data == 0, 1e-4, data)
        data[np.isnan(data)] = 0
        # print(data)
        tmp_dict['keypoint'] = data[np.newaxis, :, :, :2]
        tmp_dict['keypoint'] = np.tile(tmp_dict['keypoint'], (2, 1, 1, 1))
        # print(tmp_dict['keypoint'].shape)
        tmp_dict['keypoint_score'] = data[np.newaxis, :, :, 2]  # because we do not have class 0
        tmp_dict['keypoint_score'] = np.tile(tmp_dict['keypoint_score'], (2, 1, 1))
        # print(tmp_dict['keypoint_score'].shape)

        data = self.transform(tmp_dict)
        data = data['keypoint'][0]
        # y_onehot = torch.nn.functional.one_hot(torch.tensor(label), num_classes=5)
        # print(type(label))

        return data, label

def pre_process(data, transform):
    tmp_dict = {}
    tmp_dict['img_shape'] = (320, 480)
    tmp_dict['label'] = -1
    tmp_dict['start_index'] = 0
    tmp_dict['modality'] = 'Pose'
    tmp_dict['total_frames'] = 124
    # data = np.where(data == 0, 1e-4, data)
    data[np.isnan(data)] = 0
    # print(data)
    tmp_dict['keypoint'] = data[np.newaxis, :, :, :2]
    tmp_dict['keypoint'] = np.tile(tmp_dict['keypoint'], (2, 1, 1, 1))
    # print(tmp_dict['keypoint'].shape)
    tmp_dict['keypoint_score'] = data[np.newaxis, :, :, 2] #because we do not have class 0
    tmp_dict['keypoint_score'] = np.tile(tmp_dict['keypoint_score'], (2, 1, 1))
    # print(tmp_dict['keypoint_score'].shape)
    
    data = transform(tmp_dict)
    data = data['keypoint'][0]
    return data
        

class DataSet_Classification_Augmented(Dataset):
    def __init__(self, npy_file_path, root_dir, transform=None, data_augmentation = False):
        self.npy_data = np.load(npy_file_path)
        self.root_dir = root_dir
        self.transform = Preprocess_Module(data_augmentation)
        
    def __len__(self):
        
        # return 30
        return len(self.npy_data)
        # return 10
    
    def __getitem__(self, idx):
        label, _, video, clip = self.npy_data[idx]
        label -= 1
        data_file_path = os.path.join(self.root_dir, str(video), str(video) + '_' + str(clip) + '.npy')
        # print(data_file_path)
        data = np.load(data_file_path)
        # print(data_file_path)
        # print(data)
        #convert data to numpy
        # data = data.numpy()

        # data augmentation
        data1 = data
        data2 = data.copy()

        data2 = data2[np.newaxis, :, :, :]

        if random.random() < 0.5:
            data2 = augmentation.Shear(data2)
        if random.random() < 0.5:
            data2 = augmentation.Flip(data2)
        if random.random() < 0.5:
            data2 = augmentation.masking(data2)

        data2 = data2[0, :, :, :]
        data1 = pre_process(data1, self.transform)
        data2 = pre_process(data2, self.transform)

        data = np.concatenate((data1, data2), axis=0, dtype=np.float32)

        # print(data.shape)

        # y_onehot = torch.nn.functional.one_hot(torch.tensor(label), num_classes=5)
        # print(type(label))


        return data, label


class DataSet_Classification_osaka(Dataset):
    def __init__(self, data_list, transform=None, data_augmentation=False):
        self.npy_data = data_list
        # self.root_dir = root_dir
        self.transform = Preprocess_Module(data_augmentation)

    def __len__(self):
        # return 30

        return len(self.npy_data)
        # return 10

    def __getitem__(self, idx):
        temp_line = self.npy_data[idx].strip()
        annotations = temp_line.split(' ')
        data_file_path = annotations[0]
        label_gender = int(annotations[1])
        label_age = float(annotations[2])

        #path_nfc = unicodedata.normalize('NFC', data_file_path)
        data = np.load(data_file_path)
        tmp_dict = {}
        tmp_dict['img_shape'] = (320, 480)
        tmp_dict['label'] = -1
        tmp_dict['start_index'] = 0
        tmp_dict['modality'] = 'Pose'
        tmp_dict['total_frames'] = 124
        # data = np.where(data == 0, 1e-4, data)
        data[np.isnan(data)] = 0
        # print(data)
        tmp_dict['keypoint'] = data[np.newaxis, :, :, :2]
        tmp_dict['keypoint'] = np.tile(tmp_dict['keypoint'], (2, 1, 1, 1))
        # print(tmp_dict['keypoint'].shape)
        tmp_dict['keypoint_score'] = data[np.newaxis, :, :, 2]  # because we do not have class 0
        tmp_dict['keypoint_score'] = np.tile(tmp_dict['keypoint_score'], (2, 1, 1))
        # print(tmp_dict['keypoint_score'].shape)

        data = self.transform(tmp_dict)
        data = data['keypoint'][0]
        # y_onehot = torch.nn.functional.one_hot(torch.tensor(label), num_classes=5)
        # print(type(label))

        return data, label_gender, label_age