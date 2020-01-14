import sys
sys.path.append('model')
sys.path.append('utils')
from utils_SH import *

import numpy as np

from torch.autograd import Variable
import torch

import os

# Load Model
from defineHourglass_512_gray_skip import *

modelFolder = 'trained_model/'
my_network = HourglassNet()
my_network.load_state_dict(torch.load(os.path.join(modelFolder, 'trained_model_03.t7')))
my_network.cuda()
my_network.train(False)

from pytorch2keras import pytorch_to_keras

dummy_input = [torch.randn(1,1,512,512).cuda(), torch.rand(1,9,1,1).cuda(), torch.tensor([0])]
k_model = pytorch_to_keras(my_network, args=dummy_input)

print(k_model)