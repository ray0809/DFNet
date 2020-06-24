import os
import sys
import yaml
import time
import numpy as np
from tqdm import tqdm

import torch
import torchsnooper
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import apex
from apex import amp, optimizers

from model.dfnet import DFNet

from core.loss import *
from core.metric import *
from core.vision import VisWriter
from core.datasets import CommonDataset, DataPrefetcher
from core.utils import load_state_dict, load_auther_pretrained_weights
 
class Trainer():
  def __init__(self, cfg):
    distributed = cfg['distributed']

    self.cfg = cfg
    self.cur_step = 0
    self.cur_epoch = 0
    self.writer = None
    cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    experiment_dir = os.path.join(self.cfg['logdir'], 
                             self.cfg['dataname'], 
                             self.cfg['mask_type'], cur_time)
    if (self.cfg['global_rank'] == 0) or (not distributed):
      tensorboard_dir = os.path.join(experiment_dir, 'tensorboard')
      if cfg['tensorboard']:
          self.writer = VisWriter(logdir=tensorboard_dir)

    if (self.cfg['global_rank'] == 0) or (not distributed):    
            print('>>> use dataname: ', cfg['dataname'])
            print('>>> use input_size: ', cfg['input_size'])
            print('>>> use batch_size: ', cfg['batch_size'])
            print('>>> use num_workers: ', cfg['num_workers'])
            print('>>> use lr: ', cfg['optimizer']['lr'])
            print('>>> use mask_type: ', cfg['mask_type'])
            print('>>> tensorboard: ', cfg['tensorboard'])
            if cfg['fine_tune']:
                print('>>> finetune checkpoint: ', cfg['checkpoint'])
            
    self.train_dataset = CommonDataset(flist=cfg['flist_train'], 
                                 size=cfg['input_size'],
                                 mask_type=cfg['mask_type'],
                                 dataname=cfg['dataname'],
                                 mode='train')
    self.test_dataset = CommonDataset(flist=cfg['flist_test'], 
                                 size=cfg['input_size'],
                                 mask_type=cfg['mask_type'],
                                 dataname=cfg['dataname'],
                                 mode='test')

    self.train_sampler, self.test_sampler = None, None
    if distributed:
      self.train_sampler = DistributedSampler(self.train_dataset, 
                                        num_replicas=self.cfg['world_size'], 
                                        rank=self.cfg['global_rank'])
      self.test_sampler = DistributedSampler(self.test_dataset, 
                                        num_replicas=self.cfg['world_size'], 
                                        rank=self.cfg['global_rank'])
  
    self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                batch_size=self.cfg['batch_size'],
                                shuffle=(self.train_sampler is None),
                                num_workers=self.cfg['num_workers'],
                                sampler=self.train_sampler,
                                pin_memory=True,
                                drop_last=True)

    self.test_dataloader = DataLoader(dataset=self.test_dataset,
                                batch_size=self.cfg['batch_size'],
                                shuffle=(self.test_sampler is None),
                                num_workers=self.cfg['num_workers'],
                                sampler=self.test_sampler,
                                pin_memory=True,
                                drop_last=False)

    self.ckpt_dir = os.path.join(experiment_dir, 'ckpt')
    os.makedirs(self.ckpt_dir, exist_ok=True)

    self.net = DFNet(**self.cfg['modelParams']).cuda()
    self.optimizer = optim.Adam(self.net.parameters(), 
                                lr=self.cfg['optimizer']['lr'])  


    if cfg['fine_tune']:
          checkpoint = torch.load(cfg['checkpoint'], map_location='cuda')
          self.net.load_state_dict(checkpoint['model'])
          self.optimizer.load_state_dict(checkpoint['optim'])
          self.cur_step = checkpoint['cur_step']
          self.cur_epoch = checkpoint['cur_epoch']

    
    
    # 多卡bn合并
    if self.cfg['sync_bn']:
        self.net = apex.parallel.convert_syncbn_model(self.net)

    # 模型必须要先载入gpu才能加载混合精度
    opt_level = 'O1'  # float16
    if cfg['mix_precision']:
      self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level=opt_level)

    self.vgg = VGGFeature().cuda()
    self.style_loss = StyleLoss()
    self.perceptual_loss = PerceptualLoss()
    self.l1_loss = ReconstructionLoss()
    self.tv_loss = TotalVariationLoss()

    self.psnr = PSNR(max_val=1.0)
    self.ssim = SSIM()
    self.mae = MAE()

            
    self.scheduler = StepLR(self.optimizer, 
                            step_size=self.cfg['lr_decay_epoch'], 
                            gamma=self.cfg['lr_decay_ratio'])
    
    if distributed:
          self.net = DDP(self.net, 
                        device_ids=[cfg['global_rank']], 
                        output_device=cfg['global_rank'], 
                        broadcast_buffers=True, 
                        find_unused_parameters=False)
                        
    self.net.train()
    

  

  def train(self):
    for epoch in range(self.cur_epoch, self.cfg['epochs']):
      if self.cfg['distributed']:
        self.train_sampler.set_epoch(epoch)
      if self.cfg['global_rank'] == 0:
        tqdm_iters = tqdm(range(0, len(self.train_dataloader)))
        tqdm_iters.set_description("epoch {}/{}".format(epoch+1, self.cfg['epochs']))
      else:
        tqdm_iters = range(0, len(self.train_dataloader))
      dataprefetcher = DataPrefetcher(self.train_dataloader)
      
      for ite in tqdm_iters:
        self.cur_step += 1
        imgs, imgs_miss, masks = dataprefetcher.next()
        results, alphas, raws = self.net(imgs_miss, masks)
        loss_total, loss_dic = self.call_loss(results, imgs, masks)
              
        self.optimizer.zero_grad()
        if self.cfg['mix_precision']:
          with amp.scale_loss(loss_total, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        else:
          loss_total.backward()
        self.optimizer.step()
          
        if self.cfg['global_rank'] == 0:
          tqdm_iters.set_postfix(**loss_dic)
          if self.writer is not None:
            self.writer.summary_scalar(loss_dic, self.cur_step)


        if (self.cur_step % self.cfg['vis_img_step'] == 0) and (self.cfg['global_rank'] == 0):
          if self.writer is not None:
            self.writer.summary_img([imgs, imgs_miss, results[0], alphas[0], raws[0]],
                                    self.cur_step,
                                    title='deepfusion',
                                    lower=0)
        if (self.cur_step % self.cfg['checkpoint_step'] == 0) and (self.cfg['global_rank'] == 0):
              self.checkpoint()
      
      self.evalue()
      self.scheduler.step(self.cur_epoch) 
      self.cur_epoch += 1

  def evalue(self):
    self.net.eval()
    eval_count = 0
    batch_iters = len(self.test_dataloader)
    with torch.no_grad():
      psnr, ssim, mae = 0.0, 0.0, 0.0
      for imgs, imgs_miss, masks in self.test_dataloader:
          imgs = imgs.cuda()
          imgs_miss = imgs_miss.cuda()
          masks = masks.cuda()
          results, alphas, raws = self.net(imgs_miss, masks)
          

          i_psnr = self.psnr(imgs, results[0])
          i_ssim = self.ssim(imgs, results[0])
          i_mae = self.mae(imgs, results[0])

          
          psnr += i_psnr.item()
          ssim += i_ssim.item()
          mae += i_mae.item()
          

      psnr /= batch_iters
      ssim /= batch_iters
      mae /= batch_iters
      if self.cfg['global_rank'] == 0:
        print('\n{}/{}, psnr: {:.4f}, ssim: {:.4f}, mae: {:.4f}\n'.format(self.cur_epoch, self.cfg['epochs'], psnr, ssim, mae))
    
    self.net.train()
        

  def call_loss(self, results, imgs, masks):
    resize_masks = [resize_like(masks, result) for result in results]
    resize_imgs = [resize_like(imgs, result) for result in results]

    l1_r = [results[i] for i in self.cfg['loss']['structure_layers']]
    l1_t = [resize_imgs[i] for i in self.cfg['loss']['structure_layers']]
    l1_loss = self.l1_loss(l1_r, l1_t)
    l1_loss *= self.cfg['loss']['w_l1']
    

    vgg_r = [self.vgg(results[i]) for i in self.cfg['loss']['texture_layers']]
    vgg_t = [self.vgg(resize_imgs[i]) for i in self.cfg['loss']['texture_layers']]

    perceptual_loss = self.perceptual_loss(vgg_r, vgg_t)
    perceptual_loss *= self.cfg['loss']['w_percep']

    style_loss = self.style_loss(vgg_r, vgg_t)
    style_loss *= self.cfg['loss']['w_style']

    loss = l1_loss + perceptual_loss + style_loss
    loss_dic = {
      'l1': l1_loss.item(),
      'perceptual': perceptual_loss.item(),
      'style': style_loss.item(),
    }


    return loss, loss_dic


  def checkpoint(self):
    torch.save({'model':self.net.state_dict(),
                'cur_step':self.cur_step,
                'cur_epoch':self.cur_epoch,
                'optim':self.optimizer.state_dict(),
                },
                os.path.join(self.ckpt_dir, str(self.cur_epoch)+'.pth'))