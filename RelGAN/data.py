# Copyright (C) 2019 Willy Po-Wei Wu & Elvis Yu-Jing Lin <maya6282@gmail.com, elvisyjlin@gmail.com>
# 
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import random
import numpy as np
import skimage.io as io
import torch
import torchvision.transforms as transforms
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

#voice model 위한 데이터셋 제작 코드
class voiceModel(object):
    def __init__(self, path, image_size, selected_attrs=None,
                 filter_attrs={}, mode='train', test_num=0):
        assert mode in ['train', 'val'], 'Unsupported mode:{}'.format(mode)
        self.path = path
        self.image_size = image_size
        print("Loading annotations...")
        self.annotations, self.selected_attrs = load_annotations(join(path, 'label.txt'),selected_attrs)
        print('Loading image list...')
        self.image_list = load_image_list(os.join(path, 'image_list.txt'))
        self.filter(filter_attrs)
        if mode=='train':
                #데이터 핸들링
                self.tf = transforms.Compose([
                transforms.ToPILImage(), #tensor-> PIL 변환
                transforms.Resize(image_size), 
                transforms.CenterCrop(image_size), 
                # transforms.RandomHorizontalFlip(p=0.5), 
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        if mode == 'val':
            self.tf = transforms.Compose([
                transforms.ToPILImage(), 
                transforms.Resize(image_size), 
                transforms.CenterCrop(image_size), 
                transforms.ToTensor(), 
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        print("Splitting image list...")
        if test_num >-1:
            if mode=='train':
                print('Picking training images')
                #self.image_list = self.image_list[test_num:]
                #self.length = self.length - test_num
                
                # Pick all images as training set
                self.image_list = self.image_list
            if mode == 'val':
                print('Picking testing images')
                self.image_list = self.image_list[:test_num]
        print('Voice Trans Model dataset load.')

    def get(self, index):
        img = io.imread(os.join(self.path, str(index)+".jpg"))
        att = self.annotations[self.image_list[index]]
        return self.tf(img), torch.tensor(att)
    def __len__(self):
        return len(self.image_list)

    def filter(self, attributes):
       to_remove = []
       for img_idx, img in enumerate(self.image_list):
            for attr, val in attributes.items():
                attr_idx = self.selected_attrs.index(attr)
                if self.annotations[self.image_list[img_idx]][attr_idx] == val:
                    to_remove.append(img_idx)
                    break
            for img_idx in reversed(to_remove):
                del self.image_list[img_idx]
                del self.annotations[img_idx]
                
class wavData(object):
    def __init__(self, path, selected_attrs=None, 
                 filter_attrs={}, mode='train', val_num=500):
        assert mode in ['train', 'val'], f'Unsupported mode:{format(mode)}'
        self.path = path
        print("Loading annotations...")
        self.annotations, self.selected_attrs = load_annotations(os.path.join(path, 'label.txt'), selected_attrs)
        
        print('Loading image list...')
        #data split을 위한 list shuffle
        self.image_list = load_image_list(os.path.join(path, 'image_list.txt'))
        random.shuffle(self.image_list)
        self.filter(filter_attrs)
        self.length = len(self.image_list)

        print("Splitting image list...")
        if val_num > -1:
            if mode=='train':
                print('Picking training images')
                self.image_list = self.image_list[val_num:]
                self.length = self.length - val_num
                # Pick all images as training set
                self.image_list = self.image_list

            if mode == 'val':
                print('Picking testing images')
                self.image_list = self.image_list[:val_num]
        print('wav data Model dataset load.')

    def get(self, index):
        img = np.load(os.path.join(self.path, str(index)+".npy"))
        # att = self.annotations[self.image_list[index]]
        att = self.annotations[str(index)+".npy"]
        return torch.tensor(img), torch.tensor(att)
    def __len__(self):
        return len(self.image_list)
    
    def filter(self, attributes):
        to_remove = []
        for img_idx, img in enumerate(self.image_list):
            for attr, val in attributes.items():
                attr_idx = self.selected_attrs.index(attr)
                if self.annotations[self.image_list[img_idx]][attr_idx] == val:
                    to_remove.append(img_idx)
                    break
            for img_idx in reversed(to_remove):
                del self.image_list[img_idx]
                del self.annotations[img_idx]

class PairedData(object):
    def __init__(self, dataset, batch_size, mode):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = mode == 'train'
        self.i = 0

    def next(self, gpu=False, multi_gpu=False):
        if self.shuffle:
            idxs = np.random.choice(len(self.dataset), self.batch_size)
        else:
            idxs = list(range(self.i, self.i + self.batch_size))
            self.i = self.i + self.batch_size
            if self.i + self.batch_size >= len(self):
                self.i = 0
        
        imgs = [None] * self.batch_size
        atts = [None] * self.batch_size
        for i in range(len(idxs)):
            img, att = self.dataset.get(idxs[i])
            imgs[i] = img
            atts[i] = att
        imgs = torch.stack(imgs)
        # print("data size: ",img.size())
        atts = torch.stack(atts)
        if torch.cuda.is_available():
            imgs = imgs.to('cuda')
            atts = atts.to('cuda')
        return imgs, atts
    
    def __len__(self):
        return len(self.dataset)
class ImageNpyFolder(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes, self.class_to_idx = self._find_classes(self.root)
        self.samples = self._make_dataset(self.root, self.class_to_idx)

    def __getitem__(self, index):
        path, target = self.samples[index]
        if path.endswith('.npy'):
            sample = np.load(path)
        else:
            sample = Image.open(path).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self, dir, class_to_idx):
        images = []
        for target_class in class_to_idx.keys():
            target_dir = os.path.join(dir, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target_class])
                    images.append(item)
        return images
def load_annotations(file, selected_attrs=None):
    lines = open(file).readlines()
    '''
    angry embarrassed middle pleasure sad
    0.png    1 0 0 0 0
    1.png    1 0 0 0 0
    '''
    attrs = lines[0].split() #lines[0] = ['angry' , 'embarrassed', 'middle', 'pleasure', 'sad']
    if selected_attrs is None:
        selected_attrs = attrs
        selected_attrs_idx = [attrs.index(a) for a in selected_attrs] #0, 1 , 2, 3, 4

    annotations = {}
    for line in lines[1:]:
        tokens = line.split()
        file = tokens[0]
        # anno = [(int(t)) for t in tokens[1:]]
        # anno = [anno[idx] for idx in selected_attrs_idx]
        anno = list(map(int,tokens[1:6]))
        annotations[file] = anno
    return annotations, selected_attrs

def load_image_list(file):
    lines = open(file).readlines()
    '''
    0  0.png    1 0 0 0 0
    1  1.png    1 0 0 0 0
    '''
    image_list = [None] * len(lines)
    for line in lines:
        tokens = line.split()
        idx = int(tokens[0])
        file = tokens[1]
        image_list[idx] = file
    return image_list