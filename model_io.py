from calendar import EPOCH
import os

import torch


def save_weights(model, filename, path="./saved_models"):
    if not os.path.isdir(path):
        os.makedirs(path)

    fpath = os.path.join(path, filename)
    torch.save(model.state_dict(), fpath)
    return


# def save_checkpoint(model, optimizer, epoch, filename, root="./checkpoints"):
#     if not os.path.isdir(root):
#         os.makedirs(root)

#     fpath = os.path.join(root, filename)
#     torch.save(
#         {
#             "model": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "epoch": epoch
#         }
#         , fpath)
def save_checkpoint(args,models, optimizer, epoch, filename, root="./checkpoints"):
    # root = os.path.join(root, "weights_{}".format(epoch))
    if not os.path.isdir(root):
        os.makedirs(root)
    # save_folder = os.path.join(log_path, "models", "weights_{}".format(epoch))
    fpath = os.path.join(root,"weights_{}".format(epoch))
    # torch.save(
    #     {
    #         "model": model.state_dict(),
    #         "optimizer": optimizer.state_dict(),
    #         "epoch": epoch
    #     }
    #     , fpath)
    for model_name,model in models.items():
        save_path = os.path.join(fpath,"{}.pth".format(model_name))
        if model_name == "UnetAdaptiveBins":
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                }
                , save_path)
        
        else:
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = args.height
                to_save['width'] = args.width
                # save estimates of depth bins
                to_save['min_depth_bin'] = args.min_depth_tracker
                to_save['max_depth_bin'] = args.max_depth_tracker

            torch.save(to_save,save_path)
def load_weights(model, filename, path="./saved_models"):
    fpath = os.path.join(path, filename)
    state_dict = torch.load(fpath)
    model.load_state_dict(state_dict)
    return model


def load_checkpoint(fpath, model, optimizer=None):
    ckpt = torch.load(fpath, map_location='cpu')
    if optimizer is None:
        optimizer = ckpt.get('optimizer', None)
    else:
        optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']

    if 'model' in ckpt:
        ckpt = ckpt['model']
    load_dict = {}
    for k, v in ckpt.items():
        if k.startswith('module.'):
            k_ = k.replace('module.', '')
            load_dict[k_] = v
        else:
            load_dict[k] = v

    modified = {}  # backward compatibility to older naming of architecture blocks
    for k, v in load_dict.items():
        if k.startswith('adaptive_bins_layer.embedding_conv.'):
            k_ = k.replace('adaptive_bins_layer.embedding_conv.',
                           'adaptive_bins_layer.conv3x3.')
            modified[k_] = v
            # del load_dict[k]

        elif k.startswith('adaptive_bins_layer.patch_transformer.embedding_encoder'):

            k_ = k.replace('adaptive_bins_layer.patch_transformer.embedding_encoder',
                           'adaptive_bins_layer.patch_transformer.embedding_convPxP')
            modified[k_] = v
            # del load_dict[k]
        else:
            modified[k] = v  # else keep the original

    model.load_state_dict(modified)
    return model, optimizer, epoch
def load_pose_model(args,fpath,n,model, optimizer=None):
    assert os.path.isdir(fpath), \
            "Cannot find folder {}".format(fpath)
    print("loading model from folder {} and loading {} weights...".format(fpath,n))
    path = os.path.join(fpath, "{}.pth".format(n))
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path, map_location='cpu')
    if n == 'encoder':
        min_depth_bin = pretrained_dict.get('min_depth_bin')
        max_depth_bin = pretrained_dict.get('max_depth_bin')
        print('min depth', min_depth_bin, 'max_depth', max_depth_bin)
        if min_depth_bin is not None:
            # recompute bins
            print('setting depth bins!')
            model.compute_depth_bins(min_depth_bin, max_depth_bin)

            args.min_depth_tracker = min_depth_bin
            args.max_depth_tracker = max_depth_bin
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if optimizer is None:
        optimizer = pretrained_dict.get('optimizer', None)
    else:
        optimizer.load_state_dict(pretrained_dict['optimizer'])


    return model
    
