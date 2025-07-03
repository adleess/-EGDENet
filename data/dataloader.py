"""
Dataloader provided by the SNUNet's Authors
We adopted this dataload function directly in MSPSNet
"""
import logging
import os

import torch
import torch.utils.data as data
from PIL import Image
from tool_and_test import customedTrasforms as tr
from natsort import natsorted

# full path of the training and validation datasets
def full_path_loader(data_dir):
    train_data = [i for i in os.listdir(data_dir + '/train/time1/') if not
    i.startswith('.')]
    train_data.sort()

    valid_data = [i for i in os.listdir(data_dir + '/val/time1/') if not
    i.startswith('.')]
    # valid_data.sort()

    train_label_paths = []
    val_label_paths = []
    for img in train_data:
        train_label_paths.append(data_dir + '/train/label/' + img)
    for img in valid_data:
        val_label_paths.append(data_dir + '/val/label/' + img)


    train_data_path = []
    val_data_path = []

    for img in train_data:
        train_data_path.append([data_dir + '/train/', img])
    for img in valid_data:
        val_data_path.append([data_dir + '/val/', img])

    train_dataset = {}
    val_dataset = {}
    for cp in range(len(train_data)):
        train_dataset[cp] = {'image': train_data_path[cp],
                         'label': train_label_paths[cp]}
    for cp in range(len(valid_data)):
        val_dataset[cp] = {'image': val_data_path[cp],
                         'label': val_label_paths[cp]}


    return train_dataset, val_dataset


def full_test_loader(data_dir):

    test_data = [i for i in os.listdir(data_dir + '/val/time1/') if not i.startswith('.')]
    test_data = natsorted(test_data)

    test_label_paths = [data_dir + '/val/label/' + img for img in test_data]
    test_data_paths = [[data_dir + '/val/', img] for img in test_data]

    test_dataset = {cp: {'image': test_data_paths[cp], 'label': test_label_paths[cp]} for cp in range(len(test_data))}
    return test_dataset

def cdd_loader(img_path, label_path, aug):
    dir = img_path[0]
    name = img_path[1]

    img1 = Image.open(dir + '/time1/' + name)
    img2 = Image.open(dir + '/time2/' + name)
    label = Image.open(label_path)
    sample = {'image': (img1, img2), 'label': label}

    if aug:
        sample = tr.train_transforms(sample)
    else:
        sample = tr.test_transforms(sample)

    return sample['image'][0], sample['image'][1], sample['label']


class CDDloader(data.Dataset):

    def __init__(self, full_load, aug=False):

        self.full_load = full_load
        self.loader = cdd_loader
        self.aug = aug

    def __getitem__(self, index):

        img_path, label_path = self.full_load[index]['image'], self.full_load[index]['label']

        return self.loader(img_path,
                           label_path,
                           self.aug)

    def __len__(self):
        return len(self.full_load)



