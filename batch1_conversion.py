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
    load = 'C:/Users/ICT/Desktop/youda/AI_TeamRepo/AIModel/RelGAN/model/middletosad.19999.pth',
    save = 'model/test.pth',
    path = "C:/Users/ICT/Desktop/youda/AI_TeamRepo/test", #감정 변환된 wav파일을 저장하고 싶은 path
    steps = 40000, #GAN에서 필요함
    gpu = True,
    multi_gpu= False,
    lr = 5e-5,
    b1 = 0.5,
    b2 = 0.999,
    iterG = 50,
    iterD = 50,
    repeat_G = 6,
    batch_size = 5,
    sample_size = 2,
    l1 = 10.0,
    l2 = 10.0,
    l3 = 1.0,
    l4 = 0.000001,
    gamma = 0.1,
    zero_consistency = True,
    cycle_consistency = True,
    interpolation_regularize = True,
    orthogonal_regularize = True
)

args.selected_attributes = yaml.full_load(open(args.attr, 'r', encoding='utf-8'))
selected_attributes = args.selected_attributes
assert type(selected_attributes) is list

#=========================변경할 감정 입력하기==========================
#middle -> sad로 변경
interpolating_attributes = ['middle']

#========================변경할 데이터들 입력하기========================
img = np.load('C:/Users/ICT/Desktop/youda/AI_TeamRepo/testDatasetv15/middle/100.npy').tolist()
img = torch.tensor(img, dtype=torch.float).to('cuda')
img = img.unsqueeze(0) #입력 데이터에 맞게 numpy resize 진행하기

attr = [0, 0, 1, 0, 0]
attr = torch.tensor(attr, dtype=torch.float).to('cuda')

#kobert에서 받아올 감정 리스트
kobert_attr = [1, 0, 0, 0, 0]

#=============================모델 가져오기=============================
gan = GAN(args)
gan.to('cuda')
g, d, d_critic = gan.summary()

load_file = args.load
if load_file is not None:
    gan.load(load_file)
gan.eval()

#=============================test 시작하기=============================
attribute = torch.tensor(kobert_attr, dtype=torch.float).repeat(img.size(0),1)
attribute = attribute.cuda()
img0 = img.detach().unsqueeze(1).cpu() #음성 후처리위한 데이터?
gen = [] #generate한 데이터들 모아둘 리스트
gen.append(img0)
#Generate하기
with torch.no_grad():
    conversion = gan.G(img, attribute).detach()
gen[-1] = torch.cat([gen[-1], conversion.unsqueeze(1).cpu()], dim=1) #generate list에 쌓기
gen = torch.cat(gen)
print("00",gen.size(0), gen.size(1), gen.size(2), gen.size(3), gen.size(4))

gen = gen.view(gen.size(0)*gen.size(1)*gen.size(2), gen.size(3), gen.size(4))
print("사이즈: ",gen.size())
for i in range(gen.size(0)):
    testSet = np.split(gen, 2)
    print(f"{i}번째 사이즈: ", np.shape(testSet[i])) #0번쨰만 들어가게 수정하기'''

# np.save('C:/Users/ICT/Desktop/youda/AI_TeamRepo/test/test_4', conversion_test)

#=============================numpy data 음성으로 변환하기=============================
from inv_mel import npytomp4
for i in range(gen.size(0)):
    npytomp4(npy=testSet[i], path=os.path.join(args.path,f"test3_{i}.wav"))
