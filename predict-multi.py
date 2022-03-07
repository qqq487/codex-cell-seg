import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import argparse
import logging
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.transforms.functional as TF

from utils.data_loading import BasicDataset

from unet import UNet
from utils.utils import plot_img_and_mask

## CellPose dependency
import time, sys
import json
import pandas as pd
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl

from urllib.parse import urlparse
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import lstsq
from scipy.spatial import distance
from skimage.measure import find_contours
from skimage.morphology import disk, dilation
from scipy.ndimage.morphology import binary_dilation
from scipy.spatial.distance import cdist
from urllib.parse import urlparse

from cellpose import models, core
from cellpose import utils

from cellpose import dynamics
from scipy.ndimage import label


def stack_imgs(imgs):
    ## get first ch because in this case three channel are the same
    list_img_one_ch = [img[0] for img in imgs] 
    res = torch.stack(list_img_one_ch, dim=0)
    return res



def predict_img(net,
                full_imgs,
                device,
                scale_factor=1,
                out_threshold=0.5):
    
    ## Normalize & stack data
    norm_imgs = []
    
    for full_img in full_imgs:
        
        img_tf = TF.to_tensor(full_img)
        
        mean, std = img_tf.mean([1,2]), img_tf.std([1,2])

        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        _img = transform_norm(full_img)
        _img = _img.to(device=device, dtype=torch.float32)
        
        norm_imgs.append(_img)
        
    img = stack_imgs(norm_imgs)
    img = img.unsqueeze(0)
    
    net.eval()
    
    with torch.no_grad():
        output = net(img)
        
#         if net.n_classes > 1:
#             probs = F.softmax(output, dim=1)[0]
#         else:
#             probs = torch.sigmoid(output)[0]


#         probs = torch.sigmoid(output[:,0])
#         tf = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize((full_img.size[1], full_img.size[0])),
#             transforms.ToTensor()
#         ])
#         full_mask = tf(probs.cpu()).squeeze()
#         Y_flow = tf(output[:,1].cpu()).squeeze()
#         X_flow = tf(output[:,2].cpu()).squeeze()
        
# #         ## add predict threshold
#         thresh_func = torch.nn.Threshold(out_threshold, 0)
#         full_mask[1] = thresh_func(full_mask[1])  ## full_mask[0] = back_ground prob, full_mask[1] = cell prob
        
#         print("full_mask = ",full_mask.shape)
        
#         cv2.imwrite("probx255.png",(full_mask).numpy()*255)
        
#         cv2.imwrite("Y_flowx255.png",Y_flow.numpy()*255)
#         cv2.imwrite("X_flowx255.png",X_flow.numpy()*255)
        
        
        cellprob = output[:,0].squeeze().cpu().detach().numpy()

        dP = output[:,1:].squeeze().cpu().detach().numpy()*5 ##.transpose((2,0,1))
        
#         print("cellprob shape = ",cellprob.shape)
#         print("cellprob = ",cellprob)

#         print("dp shape = ",dP.shape)
#         print("dp = ",dP)

        masks, p, tr = dynamics.compute_masks(dP, cellprob, flow_threshold = 0)
        # cv2.imwrite("mask_5.png",(masks)*255)

    # if net.n_classes == 1:
    #     return (full_mask > out_threshold).numpy()
    # else:
    #     return (F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1)).numpy()
    
    
    return masks

    
def grow_masks(masks,growth, num_neighbors = 30):

    num_masks = len(np.unique(masks)) - 1
    
    bb_mins, bb_maxes = compute_boundbox(masks)

    print("Using sequential growth:")
    Y, X = masks.shape
    struc = disk(1)
    for _ in range(growth):
        for i in range(num_masks):
            mins = bb_mins[i]
            maxes = bb_maxes[i]
            minY, minX, maxY, maxX = mins[0] - 3*growth, mins[1] - 3*growth, maxes[0] + 3*growth, maxes[1] + 3*growth
            if minX < 0: minX = 0
            if minY < 0: minY = 0
            if maxX >= X: maxX = X - 1
            if maxY >= Y: maxY = Y - 1

            currreg = masks[minY:maxY, minX:maxX]
            mask_snippet = (currreg == i + 1)
            full_snippet = currreg > 0
            other_masks_snippet = full_snippet ^ mask_snippet
            dilated_mask = binary_dilation(mask_snippet, struc)
            final_update = (dilated_mask ^ full_snippet) ^ other_masks_snippet

            pix_to_update = np.nonzero(final_update)

            pix_X = np.array([min(j + minX, X) for j in pix_to_update[1]])
            pix_Y = np.array([min(j + minY, Y) for j in pix_to_update[0]])

            masks[pix_Y, pix_X] = i + 1

    return masks

def compute_boundbox(masks):
    
    num_masks = len(np.unique(masks)) - 1
    indices = np.where(masks != 0)
    values = masks[indices[0], indices[1]]

    maskframe = pd.DataFrame(np.transpose(np.array([indices[0], indices[1], values]))).rename(columns = {0:"y", 1:"x", 2:"id"})
    bb_mins = maskframe.groupby('id').agg({'y': 'min', 'x': 'min'}).to_records(index = False).tolist()
    bb_maxes = maskframe.groupby('id').agg({'y': 'max', 'x': 'max'}).to_records(index = False).tolist()
    
    return bb_mins, bb_maxes


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints_norm_aug_batch_4/checkpoint_epoch801.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    
    parser.add_argument('--input_folder', '-if', type=str, default="NULL", help='Filename of input folder')

    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images')
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--nuclei-stream', '-ns', default = False, help='Using Cellpose to predict nuclei and growing or not') ##
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.1,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')

    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        split = os.path.splitext(fn)
        return f'{split[0]}_OUT{split[1]}'

    return args.output or list(map(_generate_name, args.input))


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


    
    
if __name__ == '__main__':
    args = get_args()
    
    if args.input_folder != "NULL":
        in_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder)\
                    if (os.path.isdir(os.path.join(args.input_folder, f)) and ("outputs" not in f) and (not f.startswith('.')))]

        out_files = [os.path.join(args.input_folder,"outputs", f+'.png') for f in os.listdir(args.input_folder)\
                     if (os.path.isdir(os.path.join(args.input_folder, f)) and ("outputs" not in f) and (not f.startswith('.')))]
        
        print("in_files = ",in_files)
        print("out_files = ",out_files)

    else:
        in_files = args.input
        out_files = get_output_filenames(args)

    net = UNet(n_channels=5, n_classes=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        
        ## U-Net stream
        logging.info(f'\nPredicting image {filename} ...')
        
        if os.path.isdir(filename):

            img = []
            nuclei_img = []
            
            for f in os.listdir(filename):
                if not (f.startswith('.')):
                    if 'DAPI' in f:
                        nuclei_img.append(np.asarray(Image.open(os.path.join(filename,f))))
                    else:
                        img.append(Image.open(os.path.join(filename,f)))

        else:
            ## not implement yet
            img = Image.open(filename).convert('L')


        mask = predict_img(net=net,
                           full_imgs=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
        
        ## CellPose stream
        if args.nuclei_stream:
            grow_pixel = 15
            use_GPU = core.use_gpu()
            nimg = len(nuclei_img)

            model = models.CellposeModel(gpu=use_GPU, model_type='cyto',nchan = 2) ## or try model_type = 'cyto'?
            # model = models.CellposeModel(gpu=use_GPU, model_type=None) ## or try model_type = 'cyto'?
            channels = [[0,0,0]]
            masks, flows, _ = model.eval(nuclei_img, diameter=None, flow_threshold=None, channels=channels)
            masks = [_masks.astype('int32') for _masks in masks]
            growed_masks = grow_masks(masks[0], growth = grow_pixel, num_neighbors = 30)


        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)            
            result.save(out_filename)
            
            npy_name = out_filename.split('.png')[0]+'.npy'
            print("npy_name = ",npy_name)
            np.save(npy_name, mask)

            grow_npy_name = out_filename.split('.png')[0]+'grow_{}.npy'.format(grow_pixel)
            print("grow_npy_name = ",grow_npy_name)
            np.save(grow_npy_name, growed_masks)
            
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)