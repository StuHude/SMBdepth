from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image
from numpy.lib.type_check import imag  # using pillow-simd for increased speed
import torch
from torch.utils.data import Dataset, sampler,DataLoader
from torchvision import transforms
import skimage.transform

from kitti_utils import generate_depth_map

def _is_pil_image(img):
    return isinstance(img, Image.Image)
def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})
def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def preprocessing_transforms(mode):
    return transforms.Compose([
        ToTensor(mode=mode)
    ])
# loader使用torchvision中自带的transforms函数
import utils
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
loader = transforms.Compose([
     transforms.ToTensor()])
 
unloader = transforms.ToPILImage()
def save_image(tensor):
     image = tensor.cpu().clone() # we clone the tensor to not do changes on it
     image = image.squeeze(0) # remove the fake batch dimension
     image = unloader(image)

     image.save('1.jpg')
def save_depth(depth):
    depth = depth.detach().squeeze().cpu().numpy()
    pred_path = os.path.join("pred.png")
    pred = (depth).astype('uint8')
    Image.fromarray(pred.squeeze()).save(pred_path)

    viz_path = os.path.join("pred_color.png")
    viz = utils.colorize(torch.from_numpy(depth).unsqueeze(0), vmin=None, vmax=None, cmap='magma')
    viz = Image.fromarray(viz.squeeze())
    viz.save(viz_path) 
def show_from_cv(img, title=None):
     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
     plt.figure()
     plt.imshow(img)
     if title is not None:
        plt.title(title)
     plt.pause(0.001)   
class Seq2DataLoader(object):
    def __init__(self,args,mode):
        if mode == "train":
            self.training_samples = Monodataset(args,mode,transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.training_samples, args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_threads,
                                   pin_memory=True,
                                   )
        elif mode == "online_eval":
            self.testing_samples = Monodataset(args,mode,transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples,
                                   1,# args.batch_size,
                                   shuffle=False,
                                   num_workers=1,
                                   pin_memory=False,
                                   sampler = None
                                   )
        elif mode == "test":
            self.testing_samples = Monodataset(args,mode,transform=preprocessing_transforms(mode))
            self.data = DataLoader(self.testing_samples, 1, shuffle=False, num_workers=1)
        
    
class Monodataset(Dataset):
    def __init__(self,args,mode,transform=None,is_for_online_eval=False):
        super(Monodataset,self).__init__()
        self.args = args
        self.data_path = args.data_path
        # self.filenames = filenames
        self.dataset = args.dataset
        
        if mode == 'online_eval':
            with open(args.filenames_file_eval, 'r') as f:
                self.filenames = f.readlines()
        else:
            with open(args.filenames_file, 'r') as f:
                self.filenames = f.readlines()

        self.height = args.input_height
        self.width = args.input_width
        self.mono_height = args.height
        self.mono_width = args.width
        # self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        self.frame_idxs = args.frame_ids
        self.is_train = True
        self.int_x = 0
        self.int_y = 0
        img_ext = '.png' if args.png else '.jpg'
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensorM = transforms.ToTensor()  #此处需要修改 改成自定义的Totensor

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.depth = "proj_depth/groundtruth"
        self.mode = mode
        self.transform = transform
        self.to_tensorA = ToTensor
        self.is_for_online_eval = is_for_online_eval
        self.do_kb_crop = args.do_kb_crop
        self.do_color_aug = args.do_color_aug
        self.do_flip = random.random()
        #随机修改图片的光照、对比度、饱和度和色调 
        #不同的torch vision版本有不同的参数
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        # 因为monodepth有Multi-scale Estimation 所以在此处进行了resize的准备
        # self.resize = {}
        # for i in range(self.num_scales):
        #     s = 2 ** i
        #     self.resize[i] = transforms.Resize((self.height // s, self.width // s),
        #                                        interpolation=self.interp)

        self.load_depth = self.check_depth()
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        sample_path = self.filenames[index]  # 2011_09_26/2011_09_26_drive_0022_sync 473 r  '2011_09_29/2011_09_29_drive_0004_sync 225 r'
        # print(sample_path)
        line = self.filenames[index].split() # [0]2011_09_26/2011_09_26_drive_0022_sync  [1]473 [2]r
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])       # 场景中的帧号
        else:
            frame_index = 0                  # 否则是第一帧

        if len(line) == 3:                   # 视图：左 or 右
            side = line[2]
        else:
            side = None
        inputs = {}
        inputs_eval = {}
        currentImage = None
        forwardImage = None
        backwardImage = None
        random_angle = (random.random() - 0.5) * 2 * self.args.degree
        if self.mode == "train":
            for i in self.frame_idxs:
                # image_path = self.get_image_path(self,folder, frame_index, side) # 'E:\\kitti\\sync\\2011_09_30/2011_09_30_drive_0034_sync\\image_02/data\\0000000865.png' 
                inputs[("color", i)] = Image.open(self.get_image_path(folder, frame_index + i, side))
                
                # inputs[("color",i)].save(f"color_{i}_-1.jpg")
                if self.do_kb_crop is True:
                    height = inputs[("color", i)].height
                    width = inputs[("color", i)].width
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    # depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                    inputs[("color", i)] = inputs[("color", i)].crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                # if self.args.dataset == 'nyu':
                #     inputs[("color", i)].crop((43,45,608,472))
                if self.args.do_random_rotate is True:
                    inputs[("color", i)] = self.rotate_image(inputs[("color", i)], random_angle)
                    
                inputs[("color", i)] = np.asarray(inputs[("color", i)], dtype=np.float32) / 255.0
                if i == 0:
                    self.int_x = random.randint(0, inputs[("color", i)].shape[1] - self.width)
                    self.int_y = random.randint(0, inputs[("color", i)].shape[0] - self.height)
                inputs[("color", i)] = self.random_crop(inputs[("color", i)], self.height, self.width)
                inputs[("color", i)] = self.image_train_preprocess(inputs[("color", i)])
                if i == 0:
                    currentImage = inputs[("color", i)]
                elif i == -1 :
                    forwardImage = inputs[("color", i)]
                else:
                    backwardImage = inputs[("color", i)]
            if self.load_depth:
                # inputs["depth_image"] =  self.loader(self.get_depth_path(folder,frame_index,side))
                inputs["depth_image"] = Image.open(self.get_depth_path(folder,frame_index,side))

                if self.do_kb_crop is True:
                    inputs["depth_image"] = inputs["depth_image"].crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                if self.args.do_random_rotate is True:
                    inputs["depth_image"] = self.rotate_image(inputs["depth_image"], random_angle,flag=Image.NEAREST)
                inputs["depth_image"] = np.asarray(inputs["depth_image"], dtype=np.float32)
                inputs["depth_image"] = np.expand_dims(inputs["depth_image"], axis=2)
                inputs["depth_image"] = inputs["depth_image"] / 256.0
                inputs["depth_image"] = self.random_crop(inputs["depth_image"],self.height,self.width)
                inputs["depth_image"] = self.depth_train_preprocess(inputs["depth_image"])

                inputs["depth_gt"] = self.get_depth(folder, frame_index, side)  
                inputs["depth_gt"] = np.expand_dims(inputs["depth_gt"], 2)
                if self.do_kb_crop is True:
                    # height = inputs[("color", 0)].height
                    # width = inputs[("color", 0)].width
                    # top_margin = int(height - 352)
                    # left_margin = int((width - 1216) / 2)
                    inputs["depth_gt"] = inputs["depth_gt"][top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    # inputs[("color", i)] = inputs[("color", i)].crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
                if self.args.do_random_rotate is True:
                    inputs["depth_gt"] = self.rotate_image(inputs["depth_gt"], random_angle, flag=Image.NEAREST)
                # inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
                # AdaBins中写法
                # inputs["depth_gt"] = np.asarray(inputs["depth_gt"], dtype=np.float32)
                # inputs["depth_gt"] = np.expand_dims(inputs["depth_gt"], axis=2)
            # if self.args.dataset == 'nyu':
            #     inputs["depth_gt"].crop((43,45,608,472))
            if self.dataset == 'nyu':
                inputs["depth_gt"] = inputs["depth_gt"] / 1000.0
            else:
                inputs["depth_gt"] = inputs["depth_gt"] / 256.0
            
            inputs["depth_gt"] = self.depth_crop(currentImage,inputs["depth_gt"], self.args.input_height, self.args.input_width)
            inputs["depth_gt"] = self.depth_train_preprocess(inputs["depth_gt"])
            inputs["depth_gt"] = inputs["depth_gt"].transpose(2, 0, 1)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32)) 
            
            K = self.K.copy()
            K[0, :] *= self.width # // (2 ** scale)
            K[1, :] *= self.height# // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", 0)] = torch.from_numpy(K)
            inputs[("inv_K", 0)] = torch.from_numpy(inv_K)  

            K = self.K.copy()
            K[0, :] *= self.mono_width // (2 ** 2)
            K[1, :] *= self.mono_height // (2 ** 2)
            inv_K = np.linalg.pinv(K)
            inputs[("K", 2)] = torch.from_numpy(K)
            inputs[("inv_K", 2)] = torch.from_numpy(inv_K)
            sample = {
                ('color',0): currentImage,
                ('color',-1):forwardImage,
                ('color',1):backwardImage,
                'depth':inputs["depth_image"],
                'depth_gt':inputs["depth_gt"],
                'K':inputs[("K", 0)],
                'inv_K':inputs[("inv_K", 0)],
                ('K',2):inputs[("K", 2)],
                ('inv_K',2):inputs[("inv_K", 2)]
            }
        else:
            if self.mode == "online_eval":
                self.data_path = self.args.data_path_eval
            else:
                self.data_path = self.args.data_path

            for i in self.frame_idxs:
                # image_path = self.get_image_path(folder,frame_index + i, side)
                # inputs_eval[("color", i)] = np.asarray(Image.open(self.get_image_path(folder,frame_index + i, side)), dtype=np.float32) / 255.0    
                # inputs_eval[("color", i)] = inputs_eval[("color", i)].resize((self.full_res_shape))
                inputs_eval[("color", i)] = Image.open(self.get_image_path(folder,frame_index + i, side))
                inputs_eval[("color", i)] = inputs_eval[("color", i)].resize((self.full_res_shape))
                inputs_eval[("color", i)] = np.asarray(inputs_eval[("color", i)], dtype=np.float32) / 255.0   

                if self.args.do_kb_crop is True:
                    height = inputs_eval[("color", i)].shape[0]
                    width = inputs_eval[("color", i)].shape[1]
                    top_margin = int(height - 352)
                    left_margin = int((width - 1216) / 2)
                    inputs_eval[("color", i)] = inputs_eval[("color", i)][top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            if self.mode == 'online_eval':
                try:
                    inputs_eval["depth_gt"] = self.get_depth(folder, frame_index, side)
                    # inputs_eval["depth_image"] =  self.loader(self.get_depth_path(folder,frame_index,side)) 
                    inputs_eval["depth_image"] = Image.open(self.get_depth_path(folder,frame_index,side))
                    # inputs_eval["depth_gt"] = np.expand_dims(inputs_eval["depth_gt"], 0)
                    # inputs_eval["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    # print('Missing gt for {}'.format(image_path))

                if has_valid_depth:
                    inputs_eval["depth_image"] = np.asarray(inputs_eval["depth_image"], dtype=np.float32)
                    inputs_eval["depth_image"] = np.expand_dims(inputs_eval["depth_image"], axis=2)
                    inputs_eval["depth_gt"] = np.asarray(inputs_eval["depth_gt"], dtype=np.float32)
                    inputs_eval["depth_gt"] = np.expand_dims(inputs_eval["depth_gt"], axis=2)
                    if self.args.dataset == 'nyu':
                        inputs_eval["depth_gt"] = inputs_eval["depth_gt"] / 1000.0
                        inputs_eval["depth_image"] = inputs_eval["depth_image"] / 1000.0
                    else:
                        inputs_eval["depth_gt"] = inputs_eval["depth_gt"] / 256.0
                        inputs_eval["depth_image"] = inputs_eval["depth_image"] / 256.0

            if self.args.do_kb_crop is True:
                height = inputs_eval[("color", 0)].shape[0]
                width = inputs_eval[("color", 0)].shape[1]
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                if self.mode == 'online_eval' and has_valid_depth:
                    inputs_eval["depth_gt"] = inputs_eval["depth_gt"][top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
                    inputs_eval["depth_image"] = inputs_eval["depth_image"][top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
            for i in self.frame_idxs:
                if i == 0:
                    currentImage = inputs_eval[("color", i)]
                elif i == -1 :
                    forwardImage = inputs_eval[("color", i)]
                else:
                    backwardImage = inputs_eval[("color", i)]
            inputs_eval["depth_image"] = torch.from_numpy(inputs_eval["depth_image"].transpose((2, 0, 1)))
            inputs_eval["depth_gt"] = torch.from_numpy(inputs_eval["depth_gt"].transpose((2, 0, 1)))
            K = self.K.copy()
            K[0, :] *= self.width # // (2 ** scale)
            K[1, :] *= self.height# // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", 0)] = torch.from_numpy(K)
            inputs[("inv_K", 0)] = torch.from_numpy(inv_K)  
            if self.mode == 'online_eval':
                sample = {
                    ('color',0): currentImage,
                    ('color',-1):forwardImage,
                    ('color',1):backwardImage,
                    'depth':inputs_eval["depth_image"],
                    'depth_gt':inputs_eval["depth_gt"],
                    'has_valid_depth': has_valid_depth,
                    'image_path': self.get_image_path(folder,frame_index + i, side),
                    'K':inputs[("K", 0)],'inv_K':inputs[("inv_K", 0)]
                }
            else:
                sample = {('color',0): currentImage,('color',-1):forwardImage,('color',1):backwardImage,'K':inputs[("K", 0)],'inv_K':inputs[("inv_K", 0)]}
        sample.update({"frame":self.frame_idxs})
        if self.transform:
            sample = self.transform(sample)
        return sample

    # 三个等待子类实现的成员函数    
    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)


    # 获取图像的工具函数
    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext) # '0000000865.png'
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path
    def get_depth_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext) # '0000000865.png'
        depth_path = os.path.join(
            self.data_path, folder,self.depth, "image_0{}/".format(self.side_map[side]), f_str)
        return depth_path
    def get_depth(self, folder, frame_index, side):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])
        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))
        # 'E:\\kitti\\sync\\2011_09_30/2011_09_30_drive_0033_sync\\velodyne_points/data/0000001048.bin'
        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        # if do_flip:
        #     depth_gt = np.fliplr(depth_gt)

        return depth_gt
    def random_crop(self, img, height, width):

        assert img.shape[0] >= height
        assert img.shape[1] >= width
        # assert img.shape[0] == depth.shape[0]
        # assert img.shape[1] == depth.shape[1]
        x = self.int_x
        y = self.int_y
        img = img[y:y + height, x:x + width, :]
        # depth = depth[y:y + height, x:x + width, :]
        return img
    def depth_crop(self, img, depth, height, width):
        # print(depth.shape)
        # print(img.shape)
        assert img.shape[0] >= height
        assert img.shape[1] >= width 

        x = self.int_x
        y = self.int_y
        # print(x,y)
        # img = img[y:y + height, x:x + width, :]
        # print(depth.shape)
        # print(self.filenames    )
        
        depth = depth[y:y + height, x:x + width, :]
        # print(depth.shape)
        if depth.shape[0] != height or depth.shape[1] != width:
            depth = skimage.transform.resize(
                depth,(height,width),order=0,preserve_range=True,mode="constant"
            )
        assert img.shape[0] == depth.shape[0],"%d,%d"%(img.shape[0],depth.shape[0])
        assert img.shape[1] == depth.shape[1],"%d,%d"%(img.shape[1],depth.shape[1])
        return depth
    def image_train_preprocess(self, image):
        # Random flipping
        if self.do_flip > 0.5:
            image = (image[:, ::-1, :]).copy()
            # depth_gt = (depth_gt[:, ::-1, :]).copy()
        # Random gamma, brightness, color augmentation
        do_augment = random.random()
        if do_augment > 0.5:
            image = self.augment_image(image)
        return image
    def depth_train_preprocess(self, depth_gt):
        # Random flipping

        if self.do_flip > 0.5:
            # image = (image[:, ::-1, :]).copy()
            depth_gt = (depth_gt[:, ::-1, :]).copy()
        return depth_gt

    def augment_image(self, image):
        # gamma augmentation
        gamma = random.uniform(0.9, 1.1)
        image_aug = image ** gamma

        # brightness augmentation
        if self.dataset == 'nyu':
            brightness = random.uniform(0.75, 1.25)
        else:
            brightness = random.uniform(0.9, 1.1)
        image_aug = image_aug * brightness

        # color augmentation
        colors = np.random.uniform(0.9, 1.1, size=3)
        white = np.ones((image.shape[0], image.shape[1]))
        color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
        image_aug *= color_image
        image_aug = np.clip(image_aug, 0, 1)

        return image_aug
class ToTensor(object):
    def __init__(self,mode):
        self.mode = mode
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __call__(self, sample):
        frame = sample["frame"]
        depth = sample['depth']
        if self.mode == 'train':
            depth = self.to_tensor(depth)
            del sample["depth"]
            sample.update({("depth"):depth})
        for i in frame:
            image = sample['color',i]
            del sample[("color",i)]
            image = self.to_tensor(image)
            sample.update({("color",i):image})
            image = self.normalize(image)
            sample.update({("color_tensor",i):image})
            
            # depth = sample["depth_gt"]
            # depth = self.to_tensor(depth)
            # sample["depth_gt"] = depth
        del sample["frame"]
        return sample
        # if self.mode == 'test':
        #     return {'image': image}
        
        # if self.mode == 'train':
        #     return {'image': image}
        # else:
        #     has_valid_depth = sample['has_valid_depth']
        #     return {'image': image, 'has_valid_depth': has_valid_depth,
        #             'image_path': sample['image_path'], 'depth_path': sample['depth_path']}

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



