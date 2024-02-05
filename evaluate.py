import argparse
import os
from pickle import NONE
from random import randrange
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

import model_io
from dataloader import DepthDataLoader
from models import UnetAdaptiveBins
from utils import RunningAverageDict
import utils
import time
import cv2

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    note =open('eigen.txt',mode = 'a+')
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    weight_abs_rel = np.mean(np.abs(gt - pred) / (gt**2))
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    # print(f"a1:{a1} a2:{a2} a3:{a3} abs_rel:{abs_rel} rmse:{rmse} log_10:{log_10} rmse_log:{rmse_log} sq_rel:{sq_rel} silog:{silog} weight_abs_rel:{weight_abs_rel}")
    note.write(f"a1:{a1} a2:{a2} a3:{a3} abs_rel:{abs_rel} rmse:{rmse} log_10:{log_10} rmse_log:{rmse_log} sq_rel:{sq_rel} silog:{silog} weight_abs_rel:{weight_abs_rel}\n")
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel,weight_abs_rel = weight_abs_rel)


# def denormalize(x, device='cpu'):
#     mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
#     std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
#     return x * std + mean
#
def predict_tta(model, image, args):
    pred = model(image)[-1]
    #     pred = utils.depth_norm(pred)
    #     pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred = np.clip(pred.cpu().numpy(), 10, 1000)/100.
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)

    image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(device)

    pred_lr = model(image)[-1]
    #     pred_lr = utils.depth_norm(pred_lr)
    #     pred_lr = nn.functional.interpolate(pred_lr, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred_lr = np.clip(pred_lr.cpu().numpy()[...,::-1], 10, 1000)/100.
    pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], args.min_depth, args.max_depth)
    final = 0.5 * (pred + pred_lr)
    final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:], mode='bilinear', align_corners=True)
    return torch.Tensor(final)
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
     transforms.ToTensor()])
 
unloader = transforms.ToPILImage()
def save_image(tensor,dir):
    image = tensor.cpu().clone() # we clone the tensor to not do changes on it
    image = image.squeeze(0) # remove the fake batch dimension
    image = unloader(image)
    image.save(dir)
# def save_depth(tensor,dir):
#     image = tensor.cpu().clone() # we clone the tensor to not do changes on it
#     image = image.squeeze(0) # remove the fake batch dimension
#     image = unloader(image)
#     image.save(dir)    
def eval(model, test_loader, args, gpus=None, time_s=False):
    if gpus is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = gpus[0]

    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
    if args.save_color_dir is not None:
        os.makedirs(args.save_color_dir, exist_ok=True)
    if args.save_image_dir is not None:
        os.makedirs(args.save_image_dir, exist_ok=True)
    if args.save_depth_dir is not None:
        os.makedirs(args.save_depth_dir, exist_ok=True)
    if time:
        times = []

    metrics = RunningAverageDict()
    # crop_size = (471 - 45, 601 - 41)
    # bins = utils.get_bins(100)
    total_invalid = 0
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for batch in tqdm(sequential):

            image = batch['image'].to(device)
            gt = batch['depth'].to(device)
            if time:
                start = time.time()
                final = predict_tta(model, image, args)
                end = time.time()
                times.append(end - start)
            else:
                final = predict_tta(model, image, args)
            final = final.squeeze().cpu().numpy()
            img = image.unsqueeze(0).float()
            
            # final[final < args.min_depth] = args.min_depth
            # final[final > args.max_depth] = args.max_depth
            final[np.isinf(final)] = args.max_depth
            final[np.isnan(final)] = args.min_depth

            if args.save_dir is not None:
                if args.dataset == 'nyu':
                    impath = f"{batch['image_path'][0].replace('/', '__').replace('.jpg', '')}"
                    factor = 1000
                else:
                    dpath = batch['image_path'][0].split('/')
                    impath = dpath[1] + "_" + dpath[-1]
                    impath = impath.split('.')[0]
                    factor = 256

                # rgb_path = os.path.join(rgb_dir, f"{impath}.png")
                # tf.ToPILImage()(denormalize(image.squeeze().unsqueeze(0).cpu()).squeeze()).save(rgb_path)
                
                pred_path = os.path.join(args.save_dir, f"{impath}.png")
                pred = (final).astype('uint8')
                # Image.fromarray(pred.squeeze()).save(pred_path)
                if args.save_color_dir is not None:
                    viz_path = os.path.join(args.save_color_dir, f"{impath}_color.png")
                    viz = utils.colorize(torch.from_numpy(final).unsqueeze(0), vmin=1e-3, vmax=80, cmap='magma_r')
                    viz = Image.fromarray(viz)
                    viz.save(viz_path)
                
                if args.save_image_dir is not None:
                    image_path = os.path.join(args.save_image_dir, f"{impath}.png")
                    save_image(batch["image_show"],image_path)
                
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    # print("Invalid ground truth")
                    total_invalid += 1
                    continue
            # if args.save_depth_dir is not None:
            #         depth_path = os.path.join(args.save_depth_dir, f"{impath}.png")
            #         depth_image = gt.cpu().numpy()

            #         # final[final < args.min_depth] = args.min_depth
            #         # final[final > args.max_depth] = args.max_depth
            #         depth_image[np.isinf(depth_image)] = args.max_depth
            #         depth_image[np.isnan(depth_image)] = args.min_depth
            #         depth_image = (depth_image*256).astype('uint8')
            #         Image.fromarray(depth_image.squeeze()).save(depth_path)
            #         viz = utils.colorize(torch.from_numpy(depth_image).unsqueeze(0), vmin=None, vmax=None, cmap='magma')
            #         viz = Image.fromarray(viz.squeeze())
            #         save_depth_color_dir = args.save_depth_dir + "\\color"
            #         if save_depth_color_dir is not None:
            #             os.makedirs(save_depth_color_dir, exist_ok=True)
            #         save_depth_color_path = os.path.join(save_depth_color_dir, f"{impath}.png")
            #         viz.save(save_depth_color_path)
            gt = gt.squeeze().cpu().numpy()#.astype(np.uint8)
            # from matplotlib import pyplot as plt

            # plt.imshow(pred, interpolation='nearest', cmap='gray')
            # plt.savefig('decrypted1.png')
            # plt.show()
            valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)

            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
                        # gt = gt[valid_mask]
                        # final = final[valid_mask]
            # pred = (final).astype('uint8')
            gt_height, gt_width = gt.shape
            # err = np.ones([gt_height,gt_width])
            # for i in range(0,gt_height):
            #     for j in range(0,gt_width):
            #         if valid_mask[i][j] == True:
            #             err[i][j] = abs(float(gt[i][j]) - float(pred[i][j]))
            #         else:
            #             err[i][j] = 0
            
            # im = Image.fromarray(err*15)
            # im = im.convert('L')  # 这样才能转为灰度图，如果是彩色图则改L为‘RGB’
            # im.save('outfile.png')
            metrics.update(compute_errors(gt[valid_mask], final[valid_mask]))
            # metrics.update(compute_errors(gt, final))

    print(f"Total invalid: {total_invalid}")
    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")

    if time:
        avg_time = np.average(np.asarray(times))
        print("Average image computation time: {} s".format(avg_time))


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Model evaluator', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--n-bins', '--n_bins', default=256, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    # parser.add_argument('--save-dir', '--save_dir', default="test/eigen", type=str, help='Store predictions in folder')
    # parser.add_argument('--save-color-dir', '--save_color_dir', default="test/color", type=str, help='Store predictions in folder')
    # parser.add_argument('--save-image-dir', '--save_image_dir', default="test/Image_kitti", type=str, help='Store predictions in folder')
    # parser.add_argument('--save-depth-dir', '--save_depth_dir', default="test/Depth_kitti", type=str, help='Store predictions in folder')
    parser.add_argument('--save-dir', '--save_dir', default="test/eigen", type=str, help='Store predictions in folder')
    parser.add_argument('--save-color-dir', '--save_color_dir', default="test/color", type=str, help='Store predictions in folder')
    parser.add_argument('--save-image-dir', '--save_image_dir', default=None, type=str, help='Store predictions in folder')
    parser.add_argument('--save-depth-dir', '--save_depth_dir', default=None, type=str, help='Store predictions in folder')
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")

    parser.add_argument("--dataset", default='kitti', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='E:\\kitti\\sync', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='E:\\kitti\\sync', type=str,
                        help="path to dataset gt")

    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/kitti_eigen_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=376)
    parser.add_argument('--input_width', type=int, help='input width', default=1241)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')

    parser.add_argument('--data_path_eval',
                        default="E:\\kitti\\sync",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="E:\\kitti\\sync\\test",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')
    # parser.add_argument('--checkpoint_path', '--checkpoint-path', type=str, required=True,
                        # default='E:\\checkpoint\\Matrix_Evolution\\UnetAdaptiveBins_09-Jun_01-51\\UnetAdaptiveBins.pth',
                        # help="checkpoint file to use for prediction")
    parser.add_argument('--checkpoint_path', '--checkpoint-path', type=str, required=False,
                        default ='C:\\Users\\20811\Desktop\\CodeSample\\pretrained\\AdaBins_kitti.pt',
                        help="checkpoint file to use for prediction")

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=80)
    parser.add_argument('--eigen_crop', default=True,help='if set, crops according to Eigen NIPS14', action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--do_kb_crop', help='Use kitti benchmark cropping', action='store_true')
    parser.add_argument('--time', help="Calculate average image generation time", action="store_true")

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()
        
    # args = parser.parse_args()
    args.gpu = int(args.gpu) if args.gpu is not None else 0
    # args.gpu = None
    args.distributed = False
    device = torch.device('cuda:{}'.format(args.gpu))
    test = DepthDataLoader(args, 'online_eval').data
    model = UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                   norm='linear').to(device)
    model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
    model = model.eval()

    time_s = args.time

    eval(model, test, args, gpus=[device], time_s=time_s)
