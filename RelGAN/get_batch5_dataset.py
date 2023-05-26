import numpy as np
import torch
import os

class Dataset(object):
    '''입력받는 Data들을 bacth 만큼씩 묶어주는 코드'''
    def __init__(self, batch_size, path, attr):
        self.batch_size = batch_size
        self.path = path #voice data path
        self.attr = attr

    def next(self, gpu=True):
        imgs = [None] * self.batch_size
        attrs = {None} * self.batch_size

        for i in range(len(imgs)):
            imgs[i] = np.load(self.path).tolist()
            attrs[i] = self.attr
        imgs = torch.stack(imgs)
        attrs = torch.stack(attrs)
        if torch.cuda.is_available():
            imgs = imgs.to('cuda')
            attrs = attrs.to('cuda')
        return imgs, attrs
