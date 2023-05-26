import argparse
import os
import time
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

from RelGAN.helpers import run_from_ipython
from RelGAN.voice_nn import GAN
from RelGAN.ops import inv_normalize
import matplotlib.pyplot as plt

#GAN에 필요한 parameters
args = argparse.Namespace(
    attr = 'C:/Users/ICT/Desktop/youda/AI_TeamRepo/AIModel/RelGAN/attributes.yaml',
    data = 'C:/Users/ICT/Desktop/youda/AI_TeamRepo/voiceDatsaetv6',
    dataset = 'voiceDataset',
    load = '/C:/Users/ICT/Desktop/youda/AI_TeamRepo/AIModel/model/middletosad.19999.499.pth',
    save = 'model/test.pth',
    steps = 4000,
    start_step = 0,
    name = 'voice Conversion',
    gpu = False,
    multi_gpu= False,
    image_size=256,

    lr = 5e-5,
    b1 = 0.5,
    b2 = 0.999,
    iterG = 50,
    iterD = 50,
    repeat_G = 6,
    batch_size = 1,
    sample_size = 2,
    epochs = 5,
    l1 = 10.0,
    l2 = 10.0,
    l3 = 1.0,
    l4 = 0.000001,
    gamma = 0.1,
    zero_consistency = True,
    cycle_consistency = True,
    interpolation_regularize = True,
    orthogonal_regularize = True,

    log_interval = 100,
    sample_interval = 100,
    save_interval = 100,
    selected_attributes = ['sad']
)

args.selected_attributes = yaml.full_load(open(args.attr, 'r', encoding='utf-8'))
selected_attributes = args.selected_attributes
assert type(selected_attributes) is list

#========================변경할 감정 입력하기========================
#middle -> sad로 변경
interpolating_attributes = ['middle']

test_attributes = [
    ('angry', 0), ('embarrassed', 0), ('middle', 0), 
    ('pleasure', 0), ('sad', 1)
]
#====================================================================
inter_annos = np.zeros(
    (10 * len(interpolating_attributes), len(selected_attributes)), 
    dtype=np.float32
)

for i, attr in enumerate(interpolating_attributes):
    index = selected_attributes.index(attr)
    inter_annos[np.arange(10*i, 10*i+10), index] = np.linspace(0.1, 1, 10)

test_annos = np.zeros(
    (len(test_attributes), len(selected_attributes)), 
    dtype=np.float32
)
for i, (attr, value) in enumerate(test_attributes):
    index = selected_attributes.index(attr)
    test_annos[i, index] = value

#========================변경할 데이터들 입력하기========================
from RelGAN.get_batch5_dataset import Dataset
'''데이터 5개씩 묶어서 처리하기->batch size가 5'''
directory = 'C:/Users/ICT/Desktop/youda/AI_TeamRepo/testDatasetv15/middle'
voice_path = list(os.listdir(directory))

# voice = np.load('C:/Users/ICT/Desktop/youda/AI_TeamRepo/testDatasetv15/middle/100.npy')
idx = 0
test_datawset = Dataset(batch_size=args.batch_size, attr=[0., 0., 1., 0., 0.]) #return img, attr
# attr = torch.tensor([0, 0, 1, 0, 0], dtype=torch.float) #원래 음성 label

kobert_attr = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float).repeat(5, 1) #변환하고 싶은 감정 label
#======================================================================

#=============================모델 가져오기=============================
gan = GAN(args)
gan.to('cuda')
g, d, d_critic = gan.summary()

load_file = 'C:/Users/ICT/Desktop/youda/AI_TeamRepo/AIModel/RelGAN/model/middletosad.19999.pth'
if load_file is not None:
    gan.load(load_file)
gan.eval()
#======================================================================

#=============================test 시작하기=============================
attribute = kobert_attr - attrs #relative attribute만들기
attribute = attribute.repeat(voice.size(0), 1)
vec = attribute.cuda()

with torch.no_grad():
    #생성하기
    vc = gan.G(voice,vec).detach()

test = inv_normalize(vc)
np.save('C:/Users/ICT/Desktop/youda/AI_TeamRepo/test/test',test)