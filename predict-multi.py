import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

from utils.data_loading import BasicDataset
from utils.adaptive_entropy import DEE

from unet import UNet
from utils.utils import plot_img_and_mask
import torchvision.transforms.functional as TF



def stack_imgs(imgs):
    list_img_one_ch = [img[0] for img in imgs] ## get first ch because in this case three channel are the same
    
    #print("list_img_one_ch[0].size = ",list_img_one_ch[0].size())
    #print("list_img_one_ch[1].size = ",list_img_one_ch[1].size())
    
    res = torch.stack(list_img_one_ch, dim=0)
    #print("stack_res size = ", res.size())
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
        #_img = _img.unsqueeze(0) ## ????
        _img = _img.to(device=device, dtype=torch.float32)
        
        norm_imgs.append(_img)
        
    img = stack_imgs(norm_imgs)
    img = img.unsqueeze(0) ## ????
    
    net.eval()
    
    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze() 
        thresh_func = torch.nn.Threshold(out_threshold, 0)
        full_mask[1] = thresh_func(full_mask[1])  ## full_mask[0] = back_ground prob, full_mask[1] = cell prob
        
        
#         print("full_mask.size = ",full_mask.size())
#         print("full_mask = ",full_mask)
#         print("full_mask.argmax(dim=0).size() = ",full_mask.argmax(dim=0).size())

#         print("full_mask.argmax(dim=0) = ",full_mask.argmax(dim=0))
        
#         print("F.one_hot(full_mask.argmax(dim=0), net.n_classes) size = ",F.one_hot(full_mask.argmax(dim=0), net.n_classes).size())
#         print("F.one_hot(full_mask.argmax(dim=0), net.n_classes) = ",F.one_hot(full_mask.argmax(dim=0), net.n_classes))


    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        #return (full_mask > out_threshold).numpy()#
        return (F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1)).numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='checkpoints_norm_aug_batch_4/checkpoint_epoch801.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    
    parser.add_argument('--input_folder', '-if', type=str, default="NULL", help='Filename of input folder') ##

    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images') ## , required=True
    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+', help='Filenames of output images')
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
        ## in_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if os.path.dir(os.path.join(args.input_folder, f))]
        out_files = [os.path.join(args.input_folder,"outputs", f+'.png') for f in os.listdir(args.input_folder)\
                     if (os.path.isdir(os.path.join(args.input_folder, f)) and ("outputs" not in f) and (not f.startswith('.')))]
        print("in_files = ",in_files)
        print("out_files = ",out_files)

    else:
        in_files = args.input
        out_files = get_output_filenames(args)

    net = UNet(n_channels=5, n_classes=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info('Model loaded!')

    for i, filename in enumerate(in_files):
        logging.info(f'\nPredicting image {filename} ...')
        
        if os.path.isdir(filename):
            img = [Image.open(os.path.join(filename,f)) for f in os.listdir(filename) if not f.startswith('.')] ## img =  multi_ch_imgs

        else:
            ## norr implement yet
            img = Image.open(filename).convert('L')
    
#         dee = DEE()

#         img_ndarray = np.asarray(img)

#         result = dee.contrast_enhancement(img_ndarray)

#         img = Image.fromarray(result)

        mask = predict_img(net=net,
                           full_imgs=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)
    
        if not args.no_save:
            out_filename = out_files[i]
            result = mask_to_image(mask)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)