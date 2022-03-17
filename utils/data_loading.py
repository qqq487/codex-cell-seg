import logging
import os
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import cv2
from torchvision import transforms
import torchvision.transforms.functional as TF
from utils.adaptive_entropy import DEE
import cellpose.dynamics
from scipy.ndimage import label

import random


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0 , mask_suffix: str = '',):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray
    
    def cell_transform(self, image, mask , do_norm = 0):

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(150, 150))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Transform to tensor
        image = TF.to_tensor(image)
        
        if do_norm:
            mean, std = image.mean([1,2]), image.std([1,2])
            
#             if std[0] == std[1] and std[1] == std[2] and std[2] ==torch.tensor([0]):
#                 std += 0.00001

            transform_norm = transforms.Compose([
                #transforms.RandomAutocontrast()
                #transforms.ColorJitter(brightness=1, contrast=1, saturation=0, hue=0),
                transforms.Normalize(mean, std),
            ])
            image = transform_norm(image)
            
            
        mask = TF.to_tensor(mask).squeeze()

        return image, mask


    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            
            dee = DEE()
            
            img = Image.open(filename)

#             img_ndarray = np.asarray(img)

#             result = dee.contrast_enhancement(img_ndarray)

#             img = Image.fromarray(result)

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])

        assert img.size == mask.size, \
            'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img, mask = self.cell_transform(img, mask, 1)
    
        sample =  {
            'image': img,
            'mask': mask
        }
            
        return sample

    
class MultiMadalDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0 , mask_suffix: str = '_mask',):
        ## images_dir = './data/multi-org-same-ch/imgs'
        ## masks_dir = './data/multi-org-same-ch/masks'
        
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [file for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        # w, h = pil_img.size
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray
    
    def cell_transform(self, images, mask, do_norm = 0):
        
        num_imgs = len(images)
        # Random crop        
        if random.random() > 0.5:
            i, j, h, w = transforms.RandomCrop.get_params(
                images[0], output_size=(300, 200))
            
            mask = TF.crop(mask, i, j, h, w)
            for idx in range(num_imgs):
                images[idx] = TF.crop(images[idx], i, j, h, w)

        else:
            rand_size_x = random.randint(300,500)
            rand_size_y = random.randint(300,500)
            
            i, j, h, w = transforms.RandomCrop.get_params(
                images[0], output_size=(rand_size_x, rand_size_y))

            mask = TF.resized_crop(mask, i, j, h, w, size = (300, 200))      
            for idx in range(num_imgs):
                images[idx] = TF.resized_crop(images[idx], i, j, h, w, size = (300, 200))

#         # Random horizontal flipping
#         if random.random() > 0.5:
#             mask = TF.hflip(mask)
#             for idx in range(num_imgs):
#                 images[idx] = TF.hflip(images[idx])

#         # Random vertical flipping
#         if random.random() > 0.5:
#             mask = TF.vflip(mask)
#             for idx in range(num_imgs):
#                 images[idx] = TF.vflip(images[idx])
        
        # Random rotate
        angle = random.randint(0,359)
        mask = TF.rotate(mask, angle)
        for idx in range(num_imgs):
            images[idx] = TF.rotate(images[idx], angle)


        
        # Transform to tensor
        for idx in range(num_imgs):
            images[idx] = TF.to_tensor(images[idx])
                
        if do_norm:
            for idx in range(num_imgs):
            
                mean, std = images[idx].mean([1,2]), images[idx].std([1,2])

                transform_norm = transforms.Compose([
                    transforms.Normalize(mean, std),
                ])
                
                images[idx] = transform_norm(images[idx])
        
        ## Transform org_mask to []
        labeled_array, _ = label(mask)
        flow = cellpose.dynamics.labels_to_flows([labeled_array])
        flow = TF.to_tensor(flow[0].transpose((1, 2, 0))).squeeze()

        return images, flow

    def stack_imgs(self, imgs):
        list_img_one_ch = [img[0] for img in imgs] ## get first ch because in this case three channel are the same
        
        return torch.stack(list_img_one_ch, dim=0)
    

    
    @classmethod
    def load(cls, filename):
        
        if isinstance(filename, list):
            
            list_img_or_mask = []
            
            for file in filename:
            
                ext = splitext(file)[1]
                if ext in ['.npz', '.npy']:
                    list_img_or_mask.append(Image.fromarray(np.load(file)))
                elif ext in ['.pt', '.pth']:
                    list_img_or_mask.append(Image.fromarray(torch.load(file).numpy()))
                else:
                    list_img_or_mask.append(Image.open(file).convert('L'))
                    
            return list_img_or_mask
        
        else:
            ext = splitext(filename)[1]
            if ext in ['.npz', '.npy']:
                return Image.fromarray(np.load(filename))
            elif ext in ['.pt', '.pth']:
                return Image.fromarray(torch.load(filename).numpy())
            else:
                img = Image.open(filename).convert('L')
                return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = [os.path.join(self.images_dir,name,f) for f in listdir(os.path.join(self.images_dir,name)) if not f.startswith('.')]

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        ## assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file)
        
        for _img in img:
            assert _img.size == mask.size, \
                'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
            

        img, mask = self.cell_transform(img, mask, 1)
        
        ## shuffle channels
        # random.shuffle(img)
        
        stack_img = self.stack_imgs(img)
                
        sample =  {
            'image': stack_img,
            'mask': mask
        }
            
        return sample
