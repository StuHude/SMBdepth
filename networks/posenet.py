import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet_encoder import ResnetEncoder
from .pose_decoder import PoseDecoder
class Posenet(nn.Module):
    def __init__(self):
        super(Posenet,self).__init__()
        self.encoder = ResnetEncoder()