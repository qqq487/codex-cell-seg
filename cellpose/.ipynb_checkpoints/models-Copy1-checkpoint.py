import os, sys, time, shutil, tempfile, datetime, pathlib, subprocess
from pathlib import Path
import numpy as np
from tqdm import trange, tqdm
from urllib.parse import urlparse
import torch

import logging
models_logger = logging.getLogger(__name__)

from . import transforms, dynamics, utils, plot
from .core import UnetModel, assign_device, check_mkl, MXNET_ENABLED, parse_model_string
from .io import OMNI_INSTALLED

_MODEL_URL = 'https://www.cellpose.org/models'
_MODEL_DIR_ENV = os.environ.get("CELLPOSE_LOCAL_MODELS_PATH")
_MODEL_DIR_DEFAULT = pathlib.Path.home().joinpath('.cellpose', 'models')
MODEL_DIR = pathlib.Path(_MODEL_DIR_ENV) if _MODEL_DIR_ENV else _MODEL_DIR_DEFAULT
if OMNI_INSTALLED:
    MODEL_NAMES = ['cyto','nuclei','cyto2','bact','bact_omni','cyto2_omni']
else:
    MODEL_NAMES = ['cyto','nuclei','cyto2']

def model_path(model_type, model_index, use_torch):
    torch_str = 'torch' if use_torch else ''
    basename = '%s%s_%d' % (model_type, torch_str, model_index)
    return cache_model_path(basename)

def size_model_path(model_type, use_torch):
    torch_str = 'torch' if use_torch else ''
    basename = 'size_%s%s_0.npy' % (model_type, torch_str)
    return cache_model_path(basename)

def cache_model_path(basename):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f'{_MODEL_URL}/{basename}'
    cached_file = os.fspath(MODEL_DIR.joinpath(basename)) 
    if not os.path.exists(cached_file):
        models_logger.info('Downloading: "{}" to {}\n'.format(url, cached_file))
        utils.download_url_to_file(url, cached_file, progress=True)
    return cached_file

def deprecation_warning_cellprob_dist_threshold(cellprob_threshold, dist_threshold):
    models_logger.warning('cellprob_threshold and dist_threshold are being deprecated in a future release, use mask_threshold instead')
    return cellprob_threshold if cellprob_threshold is not None else dist_threshold


class CellposeModel(UnetModel):
    """

    Parameters
    -------------------

    gpu: bool (optional, default False)
        whether or not to save model to GPU, will check if GPU available
        
    pretrained_model: str or list of strings (optional, default False)
        path to pretrained cellpose model(s), if None or False, no model loaded
        
    model_type: str (optional, default None)
        'cyto'=cytoplasm model; 'nuclei'=nucleus model; if None, pretrained_model used
        
    net_avg: bool (optional, default True)
        loads the 4 built-in networks and averages them if True, loads one network if False
        
    torch: bool (optional, default True)
        use torch nn rather than mxnet
        
    diam_mean: float (optional, default 27.)
        mean 'diameter', 27. is built in value for 'cyto' model
        
    device: mxnet device (optional, default None)
        where model is saved (mx.gpu() or mx.cpu()), overrides gpu input,
        recommended if you want to use a specific GPU (e.g. mx.gpu(4))
        
    model_dir: str (optional, default None)
        overwrite the built in model directory where cellpose looks for models
    
    omni: use omnipose model (optional, default False)

    """
    
    # still need to put the omni model trained on cellpose data into the right folder with the right name with the size model 
    def __init__(self, gpu=False, pretrained_model=False, 
                    model_type=None, net_avg=True, torch=True,
                    diam_mean=30., device=None,
                    residual_on=True, style_on=True, concatenation=False,
                    nchan=2, nclasses=3, omni=False):
        if not torch:
            if not MXNET_ENABLED:
                torch = True
        self.torch = torch
        if isinstance(pretrained_model, np.ndarray):
            pretrained_model = list(pretrained_model)
        elif isinstance(pretrained_model, str):
            pretrained_model = [pretrained_model]
    
        # initialize according to arguments 
        # these are overwritten if a model requires it (bact_omni the most rectrictive)
        self.omni = omni
        self.nclasses = nclasses 
        self.diam_mean = diam_mean
        
        if model_type is not None or (pretrained_model and not os.path.exists(pretrained_model[0])):
            pretrained_model_string = model_type 
            if ~np.any([pretrained_model_string == s for s in MODEL_NAMES]): #also covers None case
                pretrained_model_string = 'cyto'
            if (pretrained_model and not os.path.exists(pretrained_model[0])):
                models_logger.warning('pretrained model has incorrect path')
            models_logger.info(f'>>{pretrained_model_string}<< model set to be used')
            
            nuclear = 'nuclei' in pretrained_model_string
            bacterial = 'bact' in pretrained_model_string
            
            if nuclear:
                self.diam_mean = 17. 

            # set omni flag to true if the name contains it
            self.omni = 'omni' in os.path.splitext(Path(pretrained_model_string).name)[0]
            
            #changed to only look for multiple files if net_avg is selected
            model_range = range(4) if net_avg else range(1)
            pretrained_model = [model_path(pretrained_model_string, j, torch) for j in model_range]
            residual_on, style_on, concatenation = True, True, False
        else:
            if pretrained_model:
                pretrained_model_string = pretrained_model[0]
                params = parse_model_string(pretrained_model_string)
                if params is not None:
                    residual_on, style_on, concatenation = params 
                self.omni = 'omni' in os.path.splitext(Path(pretrained_model_string).name)[0]
        
        # must have four classes for omnipose models
        # Note that omni can still be used independently for evaluation to 'mix and match'
        #would be better just to read from the model 
        if self.omni:
            self.nclasses = 4       

        # initialize network
        super().__init__(gpu=gpu, pretrained_model=False,
                         diam_mean=self.diam_mean, net_avg=net_avg, device=device,
                         residual_on=residual_on, style_on=style_on, concatenation=concatenation,
                         nclasses=self.nclasses, torch=self.torch, nchan=nchan)

        self.unet = True
        self.pretrained_model = pretrained_model
        if self.pretrained_model and len(self.pretrained_model)==1:
            self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
            if not self.torch:
                self.net.collect_params().grad_req = 'null'
        ostr = ['off', 'on']
        omnistr = ['','_omni'] #toggle by containing omni phrase 
        self.net_type = 'cellpose_residual_{}_style_{}_concatenation_{}{}'.format(ostr[residual_on],
                                                                                   ostr[style_on],
                                                                                   ostr[concatenation],
                                                                                   omnistr[omni]) 
    
    def eval(self, x, batch_size=8, channels=None, channel_axis=None, 
             z_axis=None, normalize=True, invert=False, 
             rescale=None, diameter=None, do_3D=False, anisotropy=None, net_avg=True, 
             augment=False, tile=True, tile_overlap=0.1,
             resample=True, interp=True, cluster=False,
             flow_threshold=0.4, mask_threshold=0.0, diam_threshold=12.,
             cellprob_threshold=None, dist_threshold=None,
             compute_masks=True, min_size=15, stitch_threshold=0.0, progress=None, omni=False, 
             calc_trace=False, verbose=False, transparency=False, loop_run=False, model_loaded=False):
        """
            segment list of images x, or 4D array - Z x nchan x Y x X

            Parameters
            ----------
            x: list or array of images
                can be list of 2D/3D/4D images, or array of 2D/3D/4D images

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            channels: list (optional, default None)
                list of channels, either of length 2 or of length number of images by 2.
                First element of list is the channel to segment (0=grayscale, 1=red, 2=green, 3=blue).
                Second element of list is the optional nuclear channel (0=none, 1=red, 2=green, 3=blue).
                For instance, to segment grayscale images, input [0,0]. To segment images with cells
                in green and nuclei in blue, input [2,3]. To segment one grayscale image and one
                image with cells in green and nuclei in blue, input [[0,0], [2,3]].

            channel_axis: int (optional, default None)
                if None, channels dimension is attempted to be automatically determined

            z_axis: int (optional, default None)
                if None, z dimension is attempted to be automatically determined

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            invert: bool (optional, default False)
                invert image pixel intensity before running network

            rescale: float (optional, default None)
                resize factor for each image, if None, set to 1.0

            diameter: float (optional, default None)
                diameter for each image (only used if rescale is None), 
                if diameter is None, set to diam_mean

            do_3D: bool (optional, default False)
                set to True to run 3D segmentation on 4D image input

            anisotropy: float (optional, default None)
                for 3D segmentation, optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)

            net_avg: bool (optional, default True)
                runs the 4 built-in networks and averages them if True, runs one network if False

            augment: bool (optional, default False)
                tiles image with overlapping tiles and flips overlapped regions to augment

            tile: bool (optional, default True)
                tiles image to ensure GPU/CPU memory usage limited (recommended)

            tile_overlap: float (optional, default 0.1)
                fraction of overlap of tiles when computing flows

            resample: bool (optional, default True)
                run dynamics at original image size (will be slower but create more accurate boundaries)

            interp: bool (optional, default True)
                interpolate during 2D dynamics (not available in 3D) 
                (in previous versions it was False)

            flow_threshold: float (optional, default 0.4)
                flow error threshold (all cells with errors below threshold are kept) (not used for 3D)

            mask_threshold: float (optional, default 0.0)
                all pixels with value above threshold kept for masks, decrease to find more and larger masks

            dist_threshold: float (optional, default None) DEPRECATED
                use mask_threshold instead

            cellprob_threshold: float (optional, default None) DEPRECATED
                use mask_threshold instead

            compute_masks: bool (optional, default True)
                Whether or not to compute dynamics and return masks.
                This is set to False when retrieving the styles for the size model.

            min_size: int (optional, default 15)
                minimum number of pixels per mask, can turn off with -1

            stitch_threshold: float (optional, default 0.0)
                if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation

            progress: pyqt progress bar (optional, default None)
                to return progress bar status to GUI
                
            omni: bool (optional, default False)
                use omnipose mask recontruction features
            
            calc_trace: bool (optional, default False)
                calculate pixel traces and return as part of the flow
                
            verbose: bool (optional, default False)
                turn on additional output to logs for debugging 
            
            transparency: bool (optional, default False)
                modulate flow opacity by magnitude instead of brightness (can use flows on any color background) 
            
            loop_run: bool (optional, default False)
                internal variable for determining if model has been loaded, stops model loading in loop over images

            model_loaded: bool (optional, default False)
                internal variable for determining if model has been loaded, used in __main__.py

            Returns
            -------
            masks: list of 2D arrays, or single 3D array (if do_3D=True)
                labelled image, where 0=no masks; 1,2,...=mask labels

            flows: list of lists 2D arrays, or list of 3D arrays (if do_3D=True)
                flows[k][0] = XY flow in HSV 0-255
                flows[k][1] = flows at each pixel
                flows[k][2] = scalar cell probability (Cellpose) or distance transform (Omnipose)
                flows[k][3] = boundary output (nonempty for Omnipose)
                flows[k][4] = final pixel locations after Euler integration 
                flows[k][5] = pixel traces (nonempty for calc_trace=True)

            styles: list of 1D arrays of length 64, or single 1D array (if do_3D=True)
                style vector summarizing each image, also used to estimate size of objects in image

        """
        
        if cellprob_threshold is not None or dist_threshold is not None:
            
            mask_threshold = deprecation_warning_cellprob_dist_threshold(cellprob_threshold, dist_threshold)
        
        if verbose:
            models_logger.info('Evaluating with flow_threshold %0.2f, mask_threshold %0.2f'%(flow_threshold, mask_threshold))
            if omni:
                models_logger.info(f'using omni model, cluster {cluster}')
        
        
        if isinstance(x, list) or x.squeeze().ndim==5:
            masks, styles, flows = [], [], []
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            nimg = len(x)
            iterator = trange(nimg, file=tqdm_out) if nimg>1 else range(nimg)
            for i in iterator:                
                maski, stylei, flowi = self.eval(x[i], 
                                                 batch_size=batch_size, 
                                                 channels=channels[i] if (len(channels)==len(x) and 
                                                                          (isinstance(channels[i], list) or isinstance(channels[i], np.ndarray)) and
                                                                          len(channels[i])==2) else channels, 
                                                 channel_axis=channel_axis, 
                                                 z_axis=z_axis, 
                                                 normalize=normalize, 
                                                 invert=invert, 
                                                 rescale=rescale[i] if isinstance(rescale, list) or isinstance(rescale, np.ndarray) else rescale,
                                                 diameter=diameter[i] if isinstance(diameter, list) or isinstance(diameter, np.ndarray) else diameter, 
                                                 do_3D=do_3D, 
                                                 anisotropy=anisotropy, 
                                                 net_avg=net_avg, 
                                                 augment=augment, 
                                                 tile=tile, 
                                                 tile_overlap=tile_overlap,
                                                 resample=resample, 
                                                 interp=interp,
                                                 cluster=cluster,
                                                 flow_threshold=flow_threshold, 
                                                 mask_threshold=mask_threshold, 
                                                 diam_threshold=diam_threshold,
                                                 compute_masks=compute_masks, 
                                                 min_size=min_size, 
                                                 stitch_threshold=stitch_threshold, 
                                                 progress=progress,
                                                 omni=omni,
                                                 calc_trace=calc_trace, 
                                                 verbose=verbose,
                                                 transparency=transparency,
                                                 loop_run=(i>0),
                                                 model_loaded=model_loaded)
                masks.append(maski)
                flows.append(flowi)
                styles.append(stylei)
            return masks, styles, flows 
        
        else:            
#             if not model_loaded and (isinstance(self.pretrained_model, list) and not net_avg and not loop_run):
#                 print("5")
                
#                 self.net.load_model(self.pretrained_model[0], cpu=(not self.gpu))
#                 if not self.torch:
#                     self.net.collect_params().grad_req = 'null'

            x = transforms.convert_image(x, channels, channel_axis=channel_axis, z_axis=z_axis,
                                         do_3D=(do_3D or stitch_threshold>0), normalize=False, invert=False, nchan=self.nchan, omni=omni)
            if x.ndim < 4:
                x = x[np.newaxis,...]
            self.batch_size = batch_size
            rescale = self.diam_mean / diameter if (rescale is None and (diameter is not None and diameter>0)) else rescale
            rescale = 1.0 if rescale is None else rescale
            
            print("self._run_cp input x.shape = ",x.shape)
            
            masks, styles, dP, cellprob, p, bd, tr = self._run_cp(x, 
                                                          compute_masks=compute_masks,
                                                          normalize=normalize,
                                                          invert=invert,
                                                          rescale=rescale, 
                                                          net_avg=net_avg, 
                                                          resample=resample,
                                                          augment=augment, 
                                                          tile=tile, 
                                                          tile_overlap=tile_overlap,
                                                          mask_threshold=mask_threshold, 
                                                          diam_threshold=diam_threshold,
                                                          flow_threshold=flow_threshold,
                                                          interp=interp,
                                                          cluster=cluster,
                                                          min_size=min_size, 
                                                          do_3D=do_3D, 
                                                          anisotropy=anisotropy,
                                                          stitch_threshold=stitch_threshold,
                                                          omni=omni,
                                                          calc_trace=calc_trace,
                                                          verbose=verbose)
            flows = [plot.dx_to_circ(dP,transparency=transparency), dP, cellprob, p, bd, tr]
            return masks, flows, styles

    def _run_cp(self, x, compute_masks=True, normalize=True, invert=False,
                rescale=1.0, net_avg=True, resample=True,
                augment=False, tile=True, tile_overlap=0.1,
                mask_threshold=0.0, diam_threshold=12., flow_threshold=0.4, min_size=15,
                interp=True, cluster=False, anisotropy=1.0, do_3D=False, stitch_threshold=0.0,
                omni=False, calc_trace=False, verbose=False):
        
        tic = time.time()
        shape = x.shape
        nimg = shape[0]        
        
        bd, tr = None, None
        if do_3D:
            
            img = np.asarray(x)
            if normalize or invert:
                img = transforms.normalize_img(img, invert=invert, omni=omni)
            yf, styles = self._run_3D(img, rsz=rescale, anisotropy=anisotropy, 
                                      net_avg=net_avg, augment=augment, tile=tile,
                                      tile_overlap=tile_overlap)
            cellprob = yf[0][-1] + yf[1][-1] + yf[2][-1] 
            dP = np.stack((yf[1][0] + yf[2][0], yf[0][0] + yf[2][1], yf[0][1] + yf[1][1]),
                          axis=0) # (dZ, dY, dX)
            del yf
        else:
            tqdm_out = utils.TqdmToLogger(models_logger, level=logging.INFO)
            iterator = trange(nimg, file=tqdm_out) if nimg>1 else range(nimg)
            styles = np.zeros((nimg, self.nbase[-1]), np.float32)
            if resample:
                dP = np.zeros((2, nimg, shape[1], shape[2]), np.float32)
                cellprob = np.zeros((nimg, shape[1], shape[2]), np.float32)
                
            else:
                dP = np.zeros((2, nimg, int(shape[1]*rescale), int(shape[2]*rescale)), np.float32)
                cellprob = np.zeros((nimg, int(shape[1]*rescale), int(shape[2]*rescale)), np.float32)
                
            for i in iterator:
                img = np.asarray(x[i])
                if normalize or invert:
                    img = transforms.normalize_img(img, invert=invert, omni=omni)
                if rescale != 1.0:
                    img = transforms.resize_image(img, rsz=rescale)
                    
                ## HERE
                yf, style = self._run_nets(img, net_avg=net_avg,
                                           augment=augment, tile=tile,
                                           tile_overlap=tile_overlap)
                
                print("yf.shape = ",yf.shape)
                
                if resample:
                    yf = transforms.resize_image(yf, shape[1], shape[2])

                cellprob[i] = yf[:,:,2]
                print("cellprob[i].shape = ",cellprob[i].shape)
                
                dP[:, i] = yf[:,:,:2].transpose((2,0,1))
                print("dP[:, i].shape = ",dP[:, i].shape)
                
                print("dP[:, i] = ", dP[:, i])

                if self.nclasses == 4:
                    if i==0:
                        bd = np.zeros_like(cellprob)
                    bd[i] = yf[:,:,3]
                styles[i] = style
            del yf, style
        styles = styles.squeeze()
        
        
        net_time = time.time() - tic
        if nimg > 1:
            models_logger.info('network run in %2.2fs'%(net_time))

        if compute_masks:
            tic=time.time()
            niter = 200 if (do_3D and not resample) else (1 / rescale * 200)
            if do_3D:
                masks, p, tr = dynamics.compute_masks(dP, cellprob, bd, niter=niter, mask_threshold=mask_threshold,
                                                      diam_threshold=diam_threshold, flow_threshold=flow_threshold,
                                                      interp=interp, do_3D=do_3D, min_size=min_size,
                                                      resize=None, verbose=verbose,
                                                      use_gpu=self.gpu, device=self.device, nclasses=self.nclasses)
            else:
                masks, p, tr = [], [], []
                resize = [shape[1], shape[2]] if not resample else None
                for i in iterator:
                    if not (omni and OMNI_INSTALLED):
                        # run cellpose compute_masks
                        outputs = dynamics.compute_masks(dP[:,i], cellprob[i], niter=niter, mask_threshold=mask_threshold,
                                                         flow_threshold=flow_threshold, interp=interp, resize=resize, verbose=verbose,
                                                         use_gpu=self.gpu, device=self.device, nclasses=self.nclasses)
                    else:
                        # run omnipose compute_masks
                        bdi = bd[i] if bd is not None else None
                        outputs = omnipose.core.compute_masks(dP[:,i], cellprob[i], bdi, niter=niter, mask_threshold=mask_threshold,
                                                                flow_threshold=flow_threshold, diam_threshold=diam_threshold, 
                                                                interp=interp, cluster=cluster, resize=resize, 
                                                                calc_trace=calc_trace, verbose=verbose,
                                                                use_gpu=self.gpu, device=self.device, nclasses=self.nclasses)
                    masks.append(outputs[0])
                    p.append(outputs[1])
                    tr.append(outputs[2])
                
                masks = np.array(masks)
                p = np.array(p)
                tr = np.array(tr)

                if stitch_threshold > 0 and nimg > 1:
                    models_logger.info(f'stitching {nimg} planes using stitch_threshold={stitch_threshold:0.3f} to make 3D masks')
                    masks = utils.stitch3D(masks, stitch_threshold=stitch_threshold)
            
            flow_time = time.time() - tic
            if nimg > 1:
                models_logger.info('masks created in %2.2fs'%(flow_time))
            masks, dP, cellprob, p = masks.squeeze(), dP.squeeze(), cellprob.squeeze(), p.squeeze()
            bd = bd.squeeze() if bd is not None else bd
        else:
            masks, p , tr = np.zeros(0), np.zeros(0), np.zeros(0) #pass back zeros if not compute_masks
        return masks, styles, dP, cellprob, p, bd, tr

        
    def loss_fn(self, lbl, y):
        """ loss function between true labels lbl and prediction y """
        if self.omni and OMNI_INSTALLED:
             #loss function for omnipose field 
            loss = omnipose.core.loss(self, lbl, y)
        else: # original loss function 
            veci = 5. * self._to_device(lbl[:,1:])
            lbl  = self._to_device(lbl[:,0]>.5)
            loss = self.criterion(y[:,:2] , veci) 
            if self.torch:
                loss /= 2.
            loss2 = self.criterion2(y[:,2] , lbl)
            loss = loss + loss2
        return loss        


    def train(self, train_data, train_labels, train_files=None, 
              test_data=None, test_labels=None, test_files=None,
              channels=None, normalize=True, 
              save_path=None, save_every=100, save_each=False,
              learning_rate=0.2, n_epochs=500, momentum=0.9, SGD=True,
              weight_decay=0.00001, batch_size=8, nimg_per_epoch=None,
              rescale=True, min_train_masks=5,
              omni=False, netstr=None):

        """ train network with images train_data 
        
            Parameters
            ------------------

            train_data: list of arrays (2D or 3D)
                images for training

            train_labels: list of arrays (2D or 3D)
                labels for train_data, where 0=no masks; 1,2,...=mask labels
                can include flows as additional images

            train_files: list of strings
                file names for images in train_data (to save flows for future runs)

            test_data: list of arrays (2D or 3D)
                images for testing

            test_labels: list of arrays (2D or 3D)
                labels for test_data, where 0=no masks; 1,2,...=mask labels; 
                can include flows as additional images
        
            test_files: list of strings
                file names for images in test_data (to save flows for future runs)

            channels: list of ints (default, None)
                channels to use for training

            normalize: bool (default, True)
                normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel

            save_path: string (default, None)
                where to save trained model, if None it is not saved

            save_every: int (default, 100)
                save network every [save_every] epochs

            learning_rate: float or list/np.ndarray (default, 0.2)
                learning rate for training, if list, must be same length as n_epochs

            n_epochs: int (default, 500)
                how many times to go through whole training set during training

            weight_decay: float (default, 0.00001)

            SGD: bool (default, True) 
                use SGD as optimization instead of RAdam

            batch_size: int (optional, default 8)
                number of 224x224 patches to run simultaneously on the GPU
                (can make smaller or bigger depending on GPU memory usage)

            nimg_per_epoch: int (optional, default None)
                minimum number of images to train on per epoch, 
                with a small training set (< 8 images) it may help to set to 8

            rescale: bool (default, True)
                whether or not to rescale images to diam_mean during training, 
                if True it assumes you will fit a size model after training or resize your images accordingly,
                if False it will try to train the model to be scale-invariant (works worse)

            min_train_masks: int (default, 5)
                minimum number of masks an image must have to use in training set

            netstr: str (default, None)
                name of network, otherwise saved with name as params + training start time

        """
        if rescale:
            models_logger.info(f'Training with rescale = {rescale:.2f}')
        train_data, train_labels, test_data, test_labels, run_test = transforms.reshape_train_test(train_data, train_labels,
                                                                                                   test_data, test_labels,
                                                                                                   channels, normalize, omni)
        # check if train_labels have flows
        # if not, flows computed, returned with labels as train_flows[i][0]
        labels_to_flows = dynamics.labels_to_flows if not (omni and OMNI_INSTALLED) else omnipose.core.labels_to_flows
        train_flows = labels_to_flows(train_labels, files=train_files, use_gpu=self.gpu, device=self.device)
        if run_test:
            test_flows = labels_to_flows(test_labels, files=test_files, use_gpu=self.gpu, device=self.device)
        else:
            test_flows = None

        nmasks = np.array([label[0].max() for label in train_flows])
        nremove = (nmasks < min_train_masks).sum()
        if nremove > 0:
            models_logger.warning(f'{nremove} train images with number of masks less than min_train_masks ({min_train_masks}), removing from train set')
            ikeep = np.nonzero(nmasks >= min_train_masks)[0]
            train_data = [train_data[i] for i in ikeep]
            train_flows = [train_flows[i] for i in ikeep]

        if channels is None:
            models_logger.warning('channels is set to None, input must therefore have nchan channels (default is 2)')
        model_path = self._train_net(train_data, train_flows, 
                                     test_data=test_data, test_labels=test_flows,
                                     save_path=save_path, save_every=save_every, save_each=save_each,
                                     learning_rate=learning_rate, n_epochs=n_epochs, 
                                     momentum=momentum, weight_decay=weight_decay, 
                                     SGD=SGD, batch_size=batch_size, nimg_per_epoch=nimg_per_epoch, 
                                     rescale=rescale, netstr=netstr)
        self.pretrained_model = model_path
        return model_path
