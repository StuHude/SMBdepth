import argparse
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
import matplotlib
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
# from dataloader import SeqDataLoader
# from dataloaderT import Monodataset,Seq2DataLoader
from dataloaderT_modify import Seq2DataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage, colorize,normalize_image
import numpy as np
from datetime import datetime as dt
from PIL import Image
import uuid
import wandb
from tqdm import tqdm
import time
import model_io
import models
import networks
import utils
from layers import *
from tensorboardX import SummaryWriter
import random
from torchvision import transforms
def depth_to_disp(depth, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    disp = (1 / depth - min_disp) / (max_disp - min_disp)
    return disp
def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    #     return img.transpose((2, 0, 1))
    return img

def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, image, target_size=(640, 480)):
        # image = image.resize(target_size)
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                'pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

unloader = transforms.ToPILImage()
def imshow(tensor, title=None):
     image = tensor.cpu().clone() # we clone the tensor to not do changes on it
     image = image.squeeze(0) # remove the fake batch dimension
     image = unloader(image)
     plt.imshow(image)
     if title is not None:
        plt.title(title)
     plt.pause(0.001) # pause a bit so that plots are updated
def depthshow(depth,titie=None):
    depth_show = depth.clone().squeeze(0).squeeze(0)
    depth_show = unloader(depth_show)
    # min_depth = 1e-3
    # max_depth = 80
    # depth_show[depth_show < min_depth] = min_depth
    # depth_show[depth_show > max_depth] = max_depth
    # depth_show[np.isinf(depth_show)] = max_depth
    # depth_show[np.isnan(depth_show)] = min_depth
    # depth_show = np.clip(depth_show.numpy(), min_depth, max_depth)
    plt.imshow(depth_show, cmap='magma_r')
    plt.show()

if __name__ == "__main__":
    img = Image.open("test_imgs/1.jpg")
