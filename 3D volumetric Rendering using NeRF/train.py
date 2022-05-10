
import os, sys
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
import kornia
import torch.nn as nn
import json
import numpy as np
import math
from PIL import Image
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import cv2
from torch import searchsorted

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from preprocessing_utils import *
from volumetric_rendering import *
from NeRF_model import *

##################################################################################################################
'''
This code is based on https://github.com/kwea123/nerf_pl/
It is a simplified version of this repo implementation on pytorch lightning.
'''
# Mean squared error loss metric for computing loss between ground truth and target image.
class MSELoss(nn.Module):

    '''
    Inputs:
            inputs : predicted RGB image.
            targets : Ground truth image.
    Outputs:
            loss : MSELoss
    '''
    
    def __init__(self):
        super(MSELoss, self).__init__()
        
    def forward(self, inputs, targets, mode = 'fine'):
        loss = torch.nn.functional.mse_loss(inputs['color_coarse'], targets) + torch.nn.functional.mse_loss(inputs['color_fine'], targets)
        return loss
    


# psnr metric between prediction and input ground truth image.
def psnr(prediction, img_gt):
    '''
    Inputs :
            inputs : predicted RGB image.
            img_gt : ground truth image.
    Outputs:
            psnr : predicted psnr value.
    '''
    mse = torch.nn.functional.mse_loss(prediction, img_gt)
    if mse == 0:
        mse = 1e-5
    return -10.0 * torch.log10(mse)


# Visualizing the depth of the image
def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
   Inputs:
          depth : depth map of the image.
    Outputs:
          x_ : final depth map.
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_




class NerfTrainer(LightningModule):

    '''
    NerfTrainer object for LightningModule.
    Inputs:
           dataset_path : dataset_path of the scene object.
           IMG_DIMENSION : image dimension
           batch_size : batch_size for training image.
           lr : learning rate.
           decay_step : decay_step for learning rate scheduler.
           decay_gamma : decay_gamma for learning rate scheduler.
    '''
    
    def __init__(self, dataset_path, IMG_DIMENSION, batch_size,lr,
                decay_step, decay_gamma):
        super(NerfTrainer, self).__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.lr = lr
        self.decay_step = decay_step
        self.decay_gamma = decay_gamma
        self.chunk_size = 32 * 1024
        self.image_dim = IMG_DIMENSION
        
        # Initializing loss
        self.loss = MSELoss()
        
        # Initializing positional encoding aaccording to the paper.
        self.xyz_encode = positional_encoding(3, 10)
        self.pose_encode = positional_encoding(3, 4)
        
        # Initializing models 
        self.coarse_model = NeRF()
        self.fine_model = NeRF()
        
        
        
    def forward(self, ray_samples):
        
        # dividing the data into chunks to avoid out of memory.
        num_rays = ray_samples.shape[0]
        render_output = {}
        for idx in range(0, num_rays, self.chunk_size):

            rendered_op= hierarchical_sampling(
                ray_samples[idx : idx + self.chunk_size],
                self.coarse_model,
                self.fine_model,
                self.xyz_encode,
                self.pose_encode,
                self.chunk_size,
                num_fine_samples = 64,
                num_samples = 64,
            )
            for k, v in rendered_op.items():
                if k not in render_output:
                    render_output[k] = [v]
                else:
                    render_output[k] += [v]
                    
        for k, v in render_output.items():
            render_output[k] = torch.cat(v, 0)
            
        return render_output
    
    def prepare_data(self):
        
        self.train_dataset = load_blender(self.dataset_path, self.image_dim,
                                         dataset_type = "train")
        self.val_dataset = load_blender(self.dataset_path, self.image_dim,
                                       dataset_type = "val")
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle = True,
                          batch_size = self.batch_size,
                          num_workers = 4,
                          pin_memory = True)
        
    
    def val_dataloader(self):
        
        return DataLoader(self.val_dataset,
                          shuffle = False,
                          batch_size = 1,
                          num_workers = 4,
                          pin_memory = True)
        
    
    def configure_optimizers(self):
        
        self.training_parameters = list(self.coarse_model.parameters()) + list(self.fine_model.parameters())
        self.optimizer = Adam(self.training_parameters, lr = self.lr,
                             eps = 1e-8)
        self.scheduler = MultiStepLR(self.optimizer, self.decay_step,
                                    self.decay_gamma)
        return [self.optimizer], [self.scheduler]
    
    
    def training_step(self, batch, batch_idx):
        
        img, rays = batch['imgs'], batch['rays']
        rendered_output = self(rays)
        #log = {'lr': get_learning_rate(self.optimizer)}
        log = {}
        log['train/loss'] = loss = self.loss(rendered_output, img[:, :3])
        with torch.no_grad():
            psnr_ = psnr(rendered_output['color_fine'], img[:, :3])
            log['train/psnr'] = psnr_
            
        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }
    
    def validation_step(self, batch, batch_idx):
        
        img_gt, rays = batch['imgs'], batch['rays']
        rays = rays.squeeze()
        img_gt = img_gt.squeeze()
        rendered_output = self(rays)
        log = {'val_loss': self.loss(rendered_output, img_gt[:, :3])}
        
        if batch_idx == 0:
            W, H = self.image_dim
            img_predicted = rendered_output['color_fine'].view(H, W, 3).cpu()
            img_predicted = img_predicted.permute(2, 0, 1)
            img_gt = img_gt[:, :3]
            img_gt_ = img_gt.view(H, W, 3).permute(2, 0, 1).cpu()
            depth = visualize_depth(rendered_output['depth_fine'].view(H, W)) 
            stack = torch.stack([img_gt_, img_predicted, depth])
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)
            
        log['val_psnr'] = psnr(rendered_output['color_fine'], img_gt[:, :3].squeeze())
        return log
    
    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
		}


####################################################################################################

if __name__ == '__main__':

    dataset_path = '../nerf_synthetic/lego'
    IMG_DIMENSION = (100, 100)
    batch_size = 1024
    lr = 5e-3
    decay_step = [2, 4, 8]
    decay_gamma = 0.5
    Nerf_trainer = NerfTrainer(dataset_path, IMG_DIMENSION, batch_size,
                     lr, decay_step, decay_gamma)
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join('./ckpts/exp1',
                                                                '{epoch:d}'),
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=5,)
    logger = TensorBoardLogger(
        save_dir="logs",
        name='exp1',
    )
    trainer = Trainer(max_epochs=16,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=None,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=torch.cuda.device_count(),
                      num_sanity_val_steps=1,
                      benchmark=True)
    trainer.fit(Nerf_trainer)
