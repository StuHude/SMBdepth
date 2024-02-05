import argparse
import os
from posixpath import split

from numpy.core.numeric import False_
file_dir = os.path.dirname(__file__)  # the directory that options.py resides in
# print(file_dir) c:/Users/20811/Desktop/modify 当前目录


# parser.add_argument("--gt_path", default='../dataset/kitti/sync/', type=str,
#                     help="path to dataset")
# parser.add_argument('--filenames_file',
#                     default="./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
#                     type=str, help='path to the filenames text file')
import torch
import torch.distributed as dist
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

import matplotlib
logging = True
PROJECT = "MDE-AdaBins-Windows-Test"
def depth_to_disp(depth, min_depth, max_depth):
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    disp = (1 / depth - min_disp) / (max_disp - min_disp)
    return disp 
# def colorize(value, vmin=10, vmax=1000, cmap='plasma'):
#     # normalize
#     vmin = value.min() if vmin is None else vmin
#     vmax = value.max() if vmax is None else vmax
#     if vmin != vmax:
#         value = (value - vmin) / (vmax - vmin)  # vmin..vmax
#     else:
#         # Avoid 0-division
#         value = value * 0.
#     # squeeze last dim if it exists
#     # value = value.squeeze(axis=0)

#     cmapper = matplotlib.cm.get_cmap(cmap)
#     value = cmapper(value, bytes=True)  # (nxmx4)

#     img = value[:, :, :3]

#     #     return img.transpose((2, 0, 1))
#     return img


def log_images(img, depth, pred, args, step):
    # depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    # pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    depth = utils.colorize(depth.unsqueeze(0), vmin=args.min_depth, vmax=args.max_depth, cmap='magma')
    pred = utils.colorize(pred.unsqueeze(0), vmin=args.min_depth, vmax=args.max_depth, cmap='magma')
    depth = Image.fromarray(depth.squeeze())
    pred = Image.fromarray(pred.squeeze())
    wandb.log(
        {
            "Input": [wandb.Image(img)],
            "GT": [wandb.Image(depth)],
            "Prediction": [wandb.Image(pred)]
        }, step=step)

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)
def is_rank_zero(args):
    return args.rank == 0
def predict_pose(args,inputs,model):
    outputs = {}
    if args.num_pose_frames == 2:
        pose_feats = {f_i: inputs["color", f_i] for f_i in args.frame_ids}
        for f_i in args.frame_ids[1:]:
            if f_i != "s":  
                if f_i < 0 :
                    pose_inputs = [pose_feats[f_i],pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0],pose_feats[f_i]]
                if args.pose_model_type == "separate_resnet":
                    pose_inputs = [model["pose_encoder"](torch.cat(pose_inputs, 1))]
                elif args.pose_model_type == "posecnn":
                    pose_inputs = torch.cat(pose_inputs, 1)

                axisangle, translation = model["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
    else:
        if args.pose_model_type in ["separate_resnet", "posecnn"]:
            pose_inputs = torch.cat(
                [inputs[("color", i)] for i in args.frame_ids if i != "s"], 1)

            if args.pose_model_type == "separate_resnet":
                pose_inputs = [model["pose_encoder"](pose_inputs)]

        axisangle, translation = model["pose"](pose_inputs)

        for i, f_i in enumerate(args.frame_ids[1:]):
            if f_i != "s":
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, i], translation[:, i])

    return outputs
def compute_reprojection_loss(args,pred,target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    # print(target.size())
    # print(pred.size())
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    if args.no_ssim:
        reprojection_loss = l1_loss
    else:
        ssim_loss = args.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss
def compute_unsupervied_loss(args,inputs,outputs):

    loss = 0
    reprojection_losses = []
    target = inputs[("color", 0)]
    for frame_id in args.frame_ids[1:]:
        pred = outputs[("color", frame_id)]
        reprojection_losses.append(compute_reprojection_loss(args, pred, target))
    reprojection_losses = torch.cat(reprojection_losses, 1)

    if not args.disable_automasking:
        identity_reprojection_losses = []
        for frame_id in args.frame_ids[1:]:
            pred = inputs[("color", frame_id)]
            identity_reprojection_losses.append(compute_reprojection_loss(args,pred, target))
        identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

        if args.avg_reprojection:
            identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
        else:
            # save both images, and do min all at once below
            identity_reprojection_loss = identity_reprojection_losses
        # identity_reprojection_loss = identity_reprojection_losses
    elif args.predictive_mask:
        # use the predicted mask
        mask = outputs["predictive_mask"]["disp", 0]
        if not args.v1_multiscale:
            mask = F.interpolate(mask, [args.input_height, args.input_width],mode="bilinear", align_corners=False)

        reprojection_losses *= mask

        # add a loss pushing mask to 1 (using nn.BCELoss for stability)
        weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
        loss += weighting_loss.mean()
    
    if args.avg_reprojection:
        reprojection_loss = reprojection_losses.mean(1, keepdim=True)
    else:
        reprojection_loss = reprojection_losses

    if not args.disable_automasking:
        # add random numbers to break ties
        identity_reprojection_loss += torch.randn(
            # identity_reprojection_loss.shape).cuda() * 0.00001
            identity_reprojection_loss.shape) * 0.00001
        combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
    else:
        combined = reprojection_loss
    if combined.shape[1] == 1:
        to_optimise = combined
    else:
        to_optimise, idxs = torch.min(combined, dim=1)

    if not args.disable_automasking:
        outputs["identity_selection/{}".format(0)] = (
            idxs > identity_reprojection_loss.shape[1] - 1).float()

    loss += to_optimise.mean()

    # outputs["predict_depth_upsample"] = nn.functional.interpolate(outputs["predict_depth"], depth.shape[-2:], mode='bilinear', align_corners=True)
    # outputs["predict_depth_upsample"] = nn.functional.interpolate(outputs["predict_depth"],(target.size()[2],target.size()[3]), mode='bilinear', align_corners=True)
    # disp = depth_to_disp(outputs["predict_depth_upsample"],args.min_depth,args.max_depth)
    # mean_disp = disp.mean(2, True).mean(3, True)
    # norm_disp = disp / (mean_disp + 1e-7)
    # smooth_loss = get_smooth_loss(norm_disp, target)

    # loss += args.disparity_smoothness * smooth_loss / (2 ** 0)
    
    return loss
def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        # val_bins = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else test_loader:
            step = 0
            # if 'has_valid_depth' in batch:
            #     if not batch['has_valid_depth']:
            #         continue
            
            for key, ipt in batch.items():
                # print(key,type(batch[key]),"\n")
                if key != "image_path":
                    batch[key] = ipt.to(device)
            img = batch[('color_tensor',0)].to(device)
            depth = batch['depth'].to(device)
            # print(depth.size()) 
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            # depth = depth.squeeze().unsqueeze(1)
            bins, pred = model['UnetAdaptiveBins'](img)
            
            outputs_eval = {}
            outputs_eval["predict_depth"] = pred
            outputs_eval["predict_depth_upsample"] = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
            outputs_eval.update(predict_pose(args,batch,model))
            
            if args.predictive_mask:
                features = model["encoder"](batch["color_tensor",0])
                outputs_eval["predictive_mask"] = model["predictive_mask"](features)

            for i,frame_id in enumerate(args.frame_ids[1:]):
                T = outputs_eval["cam_T_cam",0,frame_id]
                # from the authors of https://arxiv.org/abs/1712.00175
                if args.pose_model_type == "posecnn":
                    axisangle = outputs_eval[("axisangle", 0, frame_id)]
                    translation = outputs_eval[("translation", 0, frame_id)]
                    inv_depth = 1 / outputs_eval["predict_depth_upsample"]
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                cam_points = backproject_depth_val(outputs_eval["predict_depth_upsample"], batch["inv_K"])
                pix_coords = project_3d_val(cam_points, batch["K"], T)
                outputs_eval[("sample", frame_id)] = pix_coords
                outputs_eval[("color", frame_id)] = F.grid_sample(batch[("color", frame_id)], outputs_eval[("sample", frame_id)],padding_mode="border")
                if not args.disable_automasking:
                    outputs_eval[("color_identity", frame_id)] = \
                        batch[("color", frame_id)]

            reproject_loss = compute_unsupervied_loss(args,batch,outputs_eval)

            

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            losses = {}
            losses["reproject_loss"] = reproject_loss
            losses["l_dense"] = l_dense
            # losses["l_chamfer"] = l_chamfer
            val_si.append(l_dense.item())
            val_si.append(reproject_loss)
            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))
            step += 1
            if step % 1327 == 0:
                if "depth_gt" in batch:
                    depth_pred = outputs_eval[("predict_depth_upsample")]
                    depth_pred = torch.clamp(F.interpolate(depth_pred, [352, 704], mode="bilinear", align_corners=False), args.min_depth, args.max_depth)
                    depth_pred = depth_pred.detach()
                    depth_gt = batch["depth"]
                    mask = depth_gt > 0
                    depth_gt = depth_gt[mask]
                    depth_pred = depth_pred[mask]
                    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
                    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
                    depth_errors = compute_depth_errors(depth_gt, depth_pred)
                    depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
                    for i, metric in enumerate(depth_metric_names):
                        losses[metric] = np.array(depth_errors[i].cpu())
                    log(args,"val", batch, outputs_eval, losses,writers,step)

        return metrics.get_value(), val_si
def log(args, mode, inputs, outputs, losses,writers,step):
    """Write an event to the tensorboard events file
    """
    writer = writers[mode]
    for l, v in losses.items():
        writer.add_scalar("{}".format(l), v, step)

    for j in range(min(4, args.batch_size)):  # write a maxmimum of four images
        for s in args.scales:
            for frame_id in args.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id)][j].data, step)
                if s == 0 and frame_id != 0:
                    writer.add_image( 
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id)][j].data, step)

            writer.add_image(
                "disp_{}/{}".format(s, j),
                normalize_image(outputs["predict_depth_upsample"][j]), step)

            if args.predictive_mask:
                for f_idx, frame_id in enumerate(args.frame_ids[1:]):
                    writer.add_image(
                        "predictive_mask_{}_{}/{}".format(frame_id, s, j),
                        outputs["predictive_mask"]["predict_depth_upsample"][j, f_idx][None, ...],
                        step)

            elif not args.disable_automasking:
                writer.add_image(
                    "automask_{}/{}".format(s, j),
                    outputs["identity_selection/{}".format(s)][j][None, ...], step)
# loader使用torchvision中自带的transforms函数
from torchvision import transforms
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
if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
    parser.add_argument('--n-bins', '--n_bins', default=256, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--plr', '--poselearning-rate', default=1e-4, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")

    parser.add_argument('--bs', default=1, type=int, help='batch size')
    parser.add_argument('--validate-every', '--validate_every', default=5 , type=int, help='validation period')
    parser.add_argument('--gpu', default=1, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="UnetAdaptiveBins")
    parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                        choices=['linear', 'softmax', 'sigmoid'])
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")

    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")

    parser.add_argument("--workers", default=1, type=int, help="Number of workers for data loading")
    
    parser.add_argument("--dataset", default='kitti', type=str, help="Dataset to train on")
    parser.add_argument("--data_path", 
                            default='E:/kitti/sync/',
                            # default="/home/tangchenxiao212/kitti/kitti_data", 
                            type=str,
                            help="path to dataset")
    # parser.add_argument("--gt_path", default='E:/kitti/sync/train', type=str,
    #                     help="path to dataset")
    # parser.add_argument('--gt_path_eval', default="E:/kitti/sync/val/",
    #                     type=str, help='path to the groundtruth data for online evaluation')                    
    
    # 此处需要修改
    parser.add_argument('--data_path_eval',
                        default="E:/kitti/sync/",
                        # default="/home/tangchenxiao212/kitti/kitti_data",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--filenames_file',default="./splits/eigen_tang/train_files.txt",
                        # default="./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')
    parser.add_argument('--filenames_file_eval',default="./splits/eigen_tang/val_files.txt",
                        # default="./train_test_inputs/kitti_eigen_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')

    parser.add_argument('--filenames_file_modify',default="./splits/{}/train_files.txt",
                        # default="./train_test_inputs/kitti_eigen_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')
    parser.add_argument('--filenames_file_eval_modify',default="./splits/{}/val_files.txt",
                        # default="./train_test_inputs/kitti_eigen_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')
    # parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")
    # parser.add_argument("--data_path", default='../dataset/nyu/sync/', type=str,
    #                     help="path to dataset")
    # parser.add_argument("--gt_path", default='../dataset/nyu/sync/', type=str,
    #                     help="path to dataset")
    # parser.add_argument('--filenames_file',
    #                     default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
    #                     type=str, help='path to the filenames text file')
                                           

    parser.add_argument('--input_height', type=int, help='input height', default=352)
    parser.add_argument('--input_width', type=int, help='input width', default=704)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=0.001)

    # parser.add_argument('--do_random_rotate', default=False,
    #                     help='if set, will perform random rotation for augmentation',
    #                     action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop',default=True, help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')

    # parser.add_argument('--data_path_eval',
    #                     default="../dataset/nyu/official_splits/test/",
    #                     type=str, help='path to the data for online evaluation')
    # parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
    #                     type=str, help='path to the groundtruth data for online evaluation')
    # parser.add_argument('--filenames_file_eval',
    #                     default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
    #                     type=str, help='path to the filenames text file for online evaluation')


    # parser.add_argument('--data_path_eval',
    #                     default="E:/dataset/nyu/official_splits/test/",
    #                     type=str, help='path to the data for online evaluation')
    # parser.add_argument('--gt_path_eval', default="../dataset/nyu/official_splits/test/",
    #                     type=str, help='path to the groundtruth data for online evaluation')
    # parser.add_argument('--filenames_file_eval',
    #                     default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
    #                     type=str, help='path to the filenames text file for online evaluation')

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop',default=True, help='if set, crops according to Garg  ECCV16', action='store_true')
    #######################################################################################################
    # new argement for Unsupervised learning
    parser.add_argument("--split",type=str,help="which training split to use",
                                  choices=["eigen_tang","eigen_zhou", "eigen_full", "odom", "benchmark"],default="eigen_zhou")
    parser.add_argument("--frame_ids",nargs="+",type=int,help="frames to load",
                                 default=[0, -1, 1])
    parser.add_argument("--do_color_aug",default=False, action='store_true')
    parser.add_argument("--do_random_rotate",default=False, action='store_true')
    parser.add_argument("--png",default=True,
                                help="if set, trains from raw KITTI png files (instead of jpgs)",action="store_true")
    parser.add_argument("--pose_model_type",type=str,help="normal or shared",default="separate_resnet",
                                            choices=["posecnn", "separate_resnet"])
    parser.add_argument("--num_layers",type=int,help="number of resnet layers",default=18,
                                       choices=[18, 34, 50, 101, 152])
    parser.add_argument("--weights_init",type=str,help="pretrained or scratch",default="pretrained",
                                         choices=["pretrained", "scratch"])
    parser.add_argument("--pose_model_input",type=str,help="how many images the pose network gets",
                                             default="pairs",choices=["pairs", "all"])

    parser.add_argument("--no_ssim",default=False)

    parser.add_argument("--v1_multiscale",help="if set, uses monodepth v1 multiscale",action="store_true")
    parser.add_argument("--avg_reprojection", help="if set, uses average reprojection loss", action="store_true")
    parser.add_argument("--disable_automasking",default=False,help="if set, doesn't do auto-masking")
    parser.add_argument("--predictive_mask",help="if set, uses a predictive masking scheme as in Zhou et al",action="store_true")
    parser.add_argument("--scales",
                                nargs="+",
                                type=int,
                                help="scales used in the loss",
                                # default=[0, 1, 2, 3])
                                default=[0])
    parser.add_argument("--log_dir",type=str,
                        help="log directory",
                        default=os.path.join(os.path.expanduser("~"), "tmp"))      
    parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="adabins")
    parser.add_argument("--pretrained_path",type=str,help="the param of pretrained model path",default="./checkpoint/best") 
    parser.add_argument("--models_to_load",nargs="+",type=str,help="models to load",default=["pose_encoder", "pose"])        # "encoder", "depth",
    parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)                            
    args = parser.parse_args()
    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    args.chamfer = args.w_chamfer > 0
    args.num_input_frames = len(args.frame_ids)
    args.num_pose_frames = 2 if args.pose_model_input == "pairs" else args.num_input_frames
    args.multigpu = False
    args.log_path = os.path.join(args.log_dir, args.model_name)
    args.filenames_file_modify = args.filenames_file_modify.format(args.split)
    args.filenames_file_eval_modify = args.filenames_file_eval_modify.format(args.split)
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')

        print(args.rank)
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        
    else:
        if ngpus_per_node == 1:
            args.gpu = 0

    writers = {}
    for mode in ["train", "val"]:
        writers[mode] = SummaryWriter(os.path.join(args.log_path, mode))     

    optimizer_state_dict=None
    model = {}
    # the input of the model architecture
    model['UnetAdaptiveBins'] = models.UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,norm=args.norm)                           
    if args.pose_model_type == "separate_resnet":
        model["pose_encoder"] = networks.ResnetEncoder(
            args.num_layers,
            args.weights_init == "pretrained",
            num_input_images=args.num_pose_frames
        )
        model["pose"] = networks.PoseDecoder(
            model["pose_encoder"].num_ch_enc,num_input_features=1,num_frames_to_predict_for=2
        )
    elif args.pose_model_type == "posecnn":
        model["pose"] = networks.PoseCNN(args.num_input_frames if args.pose_model_input == "all" else 2)

    if args.predictive_mask:
        assert args.disable_automasking, \
            "When using predictive_mask, please disable automasking with --disable_automasking"
        # Our implementation of the predictive masking baseline has the the same architecture
        # as our depth decoder. We predict a separate mask for each source frame.
        model["encoder"] = networks.ResnetEncoder(args.num_layers, args.weights_init == "pretrained")
        model["predictive_mask"] = networks.DepthDecoder(model["encoder"].num_ch_enc, args.scales,num_output_channels=(len(args.frame_ids) - 1))
    # the input of models' param
    UnetAdaptiveBins_path = args.pretrained_path + "/UnetAdaptiveBins.pt"
    model['UnetAdaptiveBins'],_,epoch = model_io.load_checkpoint(UnetAdaptiveBins_path,model['UnetAdaptiveBins'])
    for n in args.models_to_load:
        model[n] = model_io.load_pose_model(args.pretrained_path,n,model[n])
        
    # device = args.gpu
    device = args.gpu
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cpu' #Windows Only
    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        for key,item in model.items():
            model[key] = model[key].to(device)
    
    experiment_name=args.name
    args.epoch = 0 #need modify epoch from model_io
    args.last_epoch = -1
    epochs = args.epochs
    lr=args.lr
    
    ####################################### data loader ################################################

    # train_loader = DepthDataLoader(args, 'train').data
    # test_loader = DepthDataLoader(args, 'online_eval').data
    train_loader = Seq2DataLoader(args, 'train').data
    test_loader = Seq2DataLoader(args, 'online_eval').data
    ################################################################################################
    h = args.input_height
    w = args.input_width

    backproject_depth = {}
    backproject_depth_val = {}
    project_3d = {}
    project_3d_val = {}

    backproject_depth = BackprojectDepth(args.batch_size, h, w)
    backproject_depth.to(device)

    project_3d = Project3D(args.batch_size, h, w)
    project_3d.to(device)

    backproject_depth_val = BackprojectDepth(1, 352, 1216)
    # backproject_depth_val = BackprojectDepth(args.batch_size, 352, 1216)
    backproject_depth_val.to(device)

    project_3d_val = Project3D(1, 352, 1216)
    # project_3d_val = Project3D(args.batch_size, 352, 1216)
    project_3d_val.to(device)
    ######################################  load  param  #########################################
    params = []
    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    name = f"{experiment_name}_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    # print(should_write)
    should_log = should_write and logging
    # should_log = False
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
            wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes)
        # for key,item in model.items():
        #     wandb.watch(model[key],log="all",log_freq=1327)

    ###################################### losses ##############################################
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None
    ################################################################################################
    for key,item in model.items():
        model[key].train()

    ###################################### Optimizer ################################################
    if args.same_lr:
        print("Using same LR")
        for key,item in model.items():
            params += list(model[key].parameters())
    else:
        print("Using diff LR")
        m = model['UnetAdaptiveBins'].module if args.multigpu else model['UnetAdaptiveBins']
        params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": lr}]
        for key,item in model.items():
            if key != 'UnetAdaptiveBins':
                params += [{"params":model[key].parameters(),"lr":args.plr}] # 仍需要测试
                # params += list(model["predictive_mask"].parameters())
    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf
    if not args.no_ssim:
        args.ssim = SSIM()
        args.ssim.to(device)
    ###################################### Scheduler ###############################################
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    if args.resume != '' and scheduler is not None:
        scheduler.step(args.epoch + 1)
    ################################################################################################
    
    # max_iter = len(train_loader) * epochs
    for epoch in range(args.epoch, epochs):
        if should_log: wandb.log({"Epoch": epoch}, step=step)
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if is_rank_zero(args) else enumerate(train_loader):
            before_op_time = time.time()
            optimizer.zero_grad()
            for key, ipt in batch.items():
                batch[key] = ipt.to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            outputs = {}
            img = batch[('color_tensor',0)].to(device)
            depth = batch['depth'].to(device)
            bin_edges, pred = model["UnetAdaptiveBins"](img)
            outputs["predict_depth"] = pred

            if args.predictive_mask:
                features = model["encoder"](batch["color_tensor",0])
                outputs["predictive_mask"] = model["predictive_mask"](features)

            outputs["predict_depth_upsample"] = nn.functional.interpolate(outputs["predict_depth"], depth.shape[-2:], mode='bilinear', align_corners=True)
            outputs.update(predict_pose(args,batch,model))
            for i,frame_id in enumerate(args.frame_ids[1:]):
                T = outputs["cam_T_cam",0,frame_id]
                # from the authors of https://arxiv.org/abs/1712.00175
                if args.pose_model_type == "posecnn":
                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]
                    inv_depth = 1 / outputs["predict_depth"]
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)
                cam_points = backproject_depth(outputs["predict_depth_upsample"], batch["inv_K"])
                pix_coords = project_3d(cam_points, batch["K"], T)
                outputs[("sample", frame_id)] = pix_coords
                outputs[("color", frame_id)] = F.grid_sample(batch[("color", frame_id)], outputs[("sample", frame_id)],padding_mode="border")
                if not args.disable_automasking:
                    outputs[("color_identity", frame_id)] = \
                        batch[("color", frame_id)]

            reproject_loss = compute_unsupervied_loss(args,batch,outputs) # losses 是一个字典

            mask = depth > args.min_depth
            l_dense = criterion_ueff(pred, depth, mask=mask.to(torch.bool), interpolate=True)

            if args.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
            else:
                l_chamfer = torch.Tensor([0]).to(img.device)
            loss = l_dense + args.w_chamfer * l_chamfer + reproject_loss
            loss = args.w_chamfer * l_chamfer + reproject_loss
            loss = reproject_loss
            losses = {}
            losses["reproject_loss"] = reproject_loss
            losses["l_dense"] = l_dense
            losses["l_chamfer"] = l_chamfer

            loss.backward()
            for key,item in model.items():
                nn.utils.clip_grad_norm_(model[key].parameters(), 0.1)  # optional

            duration = time.time() - before_op_time

            optimizer.step()
            if should_log and step % 5 == 0:
                wandb.log({f"Train/{criterion_ueff.name}": l_dense.item()}, step=step)
                wandb.log({f"Train/{criterion_bins.name}": l_chamfer.item()}, step=step)
                wandb.log({"Train/reproject_loss}": reproject_loss}, step=step)
            step += 1
            scheduler.step()
            
    ########################################################################################################

            if should_write and step % args.validate_every == 0:
                if "depth_gt" in batch:
                    depth_pred = outputs[("predict_depth_upsample")]
                    depth_pred = torch.clamp(F.interpolate(depth_pred, [352, 704], mode="bilinear", align_corners=False), args.min_depth, args.max_depth)
                    depth_pred = depth_pred.detach()
                    depth_gt = batch["depth_gt"]
                    mask = depth_gt > 0
                    depth_gt = depth_gt[mask]
                    depth_pred = depth_pred[mask]
                    depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)
                    depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
                    depth_errors = compute_depth_errors(depth_gt, depth_pred)
                    depth_metric_names = ["de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]
                    #绝对相对差 相对误差 均方根误差 RMSE（log）
                    for i, metric in enumerate(depth_metric_names):
                        losses[metric] = np.array(depth_errors[i].cpu())
                log(args,"train",batch,outputs,losses,writers,step)
                ################################# Validation loop ##################################################
                for m in model.values():
                    m.eval()
                metrics, val_si = validate(args, model, test_loader, criterion_ueff, epoch, epochs, device)

                # print("Validated: {}".format(metrics))
                if should_log:
                    depth_log = depth.cpu().clone().detach()
                    pred_log = pred.cpu().clone().detach()
                    log_images(img, depth_log, pred_log, args, step)
                    wandb.log({
                        f"Test/{criterion_ueff.name}": val_si.get_value(),
                        # f"Test/{criterion_bins.name}": val_bins.get_value()
                    }, step=step)

                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_latest",
                                                root=os.path.join('.', "checkpoints"))

                if metrics['abs_rel'] < best_loss and should_write:
                    model_io.save_checkpoint(model, optimizer, epoch, f"{experiment_name}_{run_id}_best.pt",
                                                root=os.path.join('.', "checkpoints"))
                    best_loss = metrics['abs_rel']
                
                for m in model.values():
                    m.train()
            #################################################################################################
            