import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"


import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, MultiMadalDataset
from utils.dice_score import dice_loss
from utils.losses import mIoULoss, FocalLoss, BinaryDiceLoss

from evaluate import evaluate
from unet import UNet, UNet_spatial, UNet_cat_spatial, UNet_cat_max, UNet_cat_max_spatial
from torchvision import transforms

def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              data_root: str =  './data/org/',
              ckpt_path: str = './checkpoints/',
              amp: bool = False):

    
    dir_img = Path(os.path.join(data_root,'imgs'))
    dir_mask = Path(os.path.join(data_root,'masks'))
    dir_checkpoint = Path(ckpt_path)
    
    # 1. Create dataset
    dataset = MultiMadalDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    criterion_lbl_bce = nn.BCEWithLogitsLoss(reduction='mean')
    criterion_lbl_dice = BinaryDiceLoss()
    criterion_lbl =  mIoULoss(n_classes=2).to(device = device)# FocalLoss()#nn.BCEWithLogitsLoss(reduction='mean') ## nn.CrossEntropyLoss()
    
    criterion_vec = nn.MSELoss(reduction='mean').to(device = device)
    global_step = 0

    # 5. Begin training
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.in_channels, \
                    f'Network has been defined with {net.in_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                                        
                    lbl_loss_bce = criterion_lbl_bce(masks_pred[:,0], true_masks[:,1])
                    lbl_loss_dice = criterion_lbl_dice(masks_pred[:,0], true_masks[:,1])

                    #lbl_loss_dice = dice_loss(masks_pred[:,0].permute(1, 0, 2),true_masks[:,1].permute(1, 0, 2), multiclass=True)
                    # lbl_loss = criterion_lbl(masks_pred[:,0], true_masks[:,1])
                    vec_loss = criterion_vec(masks_pred[:,1:], true_masks[:,2:])
                    
                    # print("lbl_loss_bce = {} lbl_loss_dice = {} lbl_loss = {} vec_loss = {}".format(lbl_loss_bce.item(), lbl_loss_dice.item(),lbl_loss.item(),vec_loss.item()))
                    
                    loss = lbl_loss_dice*0.7 + lbl_loss_bce*0.3 + vec_loss 
                    # loss = lbl_loss_dice + vec_loss
                    # loss = lbl_loss_bce + vec_loss

                    # loss = criterion_lbl(masks_pred[:,0], true_masks[:,1]) \
                    #        + 5*criterion_vec(masks_pred[:,1:], true_masks[:,2:])
                           # + dice_loss(F.softmax(masks_pred, dim=1).float(),
                           #             F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                           #             multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                #division_step = (n_train // (10 * batch_size))
                division_step = 0
                
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'true': wandb.Image(true_masks[0].float().cpu()),
                                'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        if save_checkpoint and (epoch%1000 == 0):
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1000, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=4, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.000001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')    
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=20.0,
                        help='Percent of the data that is used as validation (0-100)')
    
    parser.add_argument('--data_root', '-dr', type=str, default='./data/org/', help='training data root')  ##
    parser.add_argument('--ckpt_path', '-cp', type=str, default='./checkpoints/', help='saving checkpoints path') ##   
    
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #net = NestedUNet(n_channels = 3 , n_classes = 2 )
    net = UNet_cat_max(in_channels=5, out_channels=3, bilinear=True)

    logging.info(f'Network:\n'
                 f'\t{net.in_channels} input channels\n'
                 f'\t{net.out_channels} output channels\n')
                 #f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  data_root = args.data_root,
                  ckpt_path = args.ckpt_path,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
