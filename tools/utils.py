# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   1970-01-01 01:00:00
# @Last Modified by:   Your name
# @Last Modified time: 2022-04-07 15:51:18
#
# Created on Wed Oct 20 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2021 MeTrICS Lab
#

import nibabel as nb
import os
import torch
import sys
from datetime import datetime

sys.path.append('./')
sys.path.append('./tools')

from tools.dataloader import loader_tfmri_clip

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from warmup_scheduler import GradualWarmupScheduler

def logging_clip(config):

    if config['MODEL']=='clip-audio-video':
        folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],
                                                                                config['data']['modality'])
    else:
        folder_to_save_model = config['logging']['folder_to_save_model'].format(config['data']['path_to_workdir'],config['data']['dataset'],
                                                                                config['data']['modality'],
                                                                                config['mesh_resolution']['ico_grid'],'template')
        

    if config['augmentation']['prob_augmentation']:
        folder_to_save_model = os.path.join(folder_to_save_model,'augmentation')
    else:
        folder_to_save_model = os.path.join(folder_to_save_model,'no_augmentation')

    date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

    folder_to_save_model = os.path.join(folder_to_save_model,date)
    
    if config['MODEL']=='clip-audio' or config['MODEL']=='clip-video':

        if config['transformer']['dim'] == 192:
            folder_to_save_model = folder_to_save_model + '-tiny'
        elif config['transformer']['dim'] == 384:
            folder_to_save_model = folder_to_save_model + '-small'
        elif config['transformer']['dim'] == 768:
            folder_to_save_model = folder_to_save_model + '-base'
        elif config['transformer']['dim'] == 96:
            folder_to_save_model = folder_to_save_model + '-very-tiny'
        elif config['transformer']['dim'] == 48:
            folder_to_save_model = folder_to_save_model + '-ultra-tiny'
        

        if config['transformer']['init_weights']!=False:
            folder_to_save_model = folder_to_save_model + '-'+config['transformer']['init_weights']

        if config['clip']['fmri_encoder_trainable']:
            folder_to_save_model = folder_to_save_model + '-finetune'
        else:
            folder_to_save_model = folder_to_save_model + '-freeze'

        folder_to_save_model = folder_to_save_model + '-lag-{}'.format(config['fMRI']['temporal_lag'])

        folder_to_save_model = folder_to_save_model + '-bs-{}'.format(config['training']['bs'])

    return folder_to_save_model


def get_data_path(config):

    dataset = config['data']['dataset']
    dataloader = config['data']['dataloader']
    modality = config['data']['modality']
    #sampling = config['mesh_resolution']['sampling'] ## remove
    #registration = config['data']['registration']

    if str(dataloader) == 'bold':
        if dataset == 'HCP':
            if modality == 'tfMRI' or modality == '7T_tfMRI':
                data_path = os.path.join(config['data']['path_to_data'],'HCP','movie_frames')
            elif modality == 'rfMRI' or 'smooth_rfMRI':
                data_path = os.path.join(config['data']['path_to_data'],'HCP','rest_frames')

    else:
        raise('not implemented yet')
    
    return data_path


def get_dataloaders_clip_distributed(config, 
                    data_path,
                    world_size,
                    rank):

    dataloader = config['data']['dataloader']
    modality = config['data']['modality']

    if str(dataloader)=='bold':
        print('loading functional data')
        if modality == 'tfMRI' and (config['MODEL']=='clip-video' or config['MODEL']=='clip-audio' or config['MODEL']=='clip-fmri-video-audio'):
            train_loader, val_loader, train_sampler, val_sampler = loader_tfmri_clip(data_path,config,world_size,rank)
        return train_loader, val_loader, train_sampler, val_sampler
    else:
        raise('not implemented yet')
    
def get_dimensions(config):

    modality = config['data']['modality']
    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['ico_{}_grid'.format(ico_grid)]['num_patches']
    num_vertices = config['ico_{}_grid'.format(ico_grid)]['num_vertices']

    if config['MODEL'] in ['sit']:    
        channels = config['transformer']['channels']
    elif (config['MODEL'] in ['clip-video', 'clip-audio','clip-fmri-video-audio']) and (config['clip']['fmri_encoder'] in ['sit']):    
        channels = config['transformer']['channels']
    elif config['MODEL']== 'spherical-unet':
        channels = config['spherical-unet']['channels']
    elif config['MODEL']== 'monet':
        channels = config['monet']['channels']
    num_channels = len(channels)

    if config['MODEL'] in ['clip-video','clip-audio','clip-fmri-video-audio','sit'] and ('3T_tfMRI' in modality): 

        if config['training']['init_weights'] == 'ssl_svmae' and config['fMRI']['temporal_rep']=='concat':

            T = num_channels
            N = num_patches * config['fMRI']['nbr_frames']
            V = num_vertices
        else:
            T = num_channels
            N = num_patches
            
            V = num_vertices            

        use_bottleneck = config['transformer']['use_bottleneck']
        bottleneck_dropout = config['transformer']['bottleneck_dropout']

        print('Number of channels {}; Number of patches {}; Number of vertices {}'.format(T, N, V))
        print('Using bottleneck {}; Dropout bottleneck {}'.format(use_bottleneck,bottleneck_dropout))
        print('')

        return T, N, V, use_bottleneck, bottleneck_dropout

    if config['MODEL'] in ['clip-video','clip-audio','clip-fmri-video-audio','sit'] and (modality =='rfMRI' or modality =='tfMRI' or modality == 'smooth_rfMRI' or modality=='7T_tfMRI' in modality): 

        if config['fMRI']['temporal_rep']=='concat':

            T = num_channels
            N = num_patches * config['fMRI']['nbr_frames_ckpt']
            V = num_vertices
        elif config['fMRI']['temporal_rep']=='channels':
            T = num_channels * config['fMRI']['nbr_frames']
            N = num_patches 
            V = num_vertices
        else:
            T = num_channels
            N = num_patches
            
            V = num_vertices            

        use_bottleneck = config['transformer']['use_bottleneck']
        bottleneck_dropout = config['transformer']['bottleneck_dropout']

        print('Number of channels {}; Number of patches {}; Number of vertices {}'.format(T, N, V))
        print('Using bottleneck {}; Dropout bottleneck {}'.format(use_bottleneck,bottleneck_dropout))
        print('')

        return T, N, V, use_bottleneck, bottleneck_dropout
    
    elif config['MODEL'] in ['sit'] and (modality == 'cortical_metrics' ):

        T = num_channels
        N = num_patches
        V = num_vertices

        use_bottleneck = config['transformer']['use_bottleneck']
        bottleneck_dropout = config['transformer']['bottleneck_dropout']

        print('Number of channels {}; Number of patches {}; Number of vertices {}'.format(T, N, V))
        print('Using bottleneck {}; Dropout bottleneck {}'.format(use_bottleneck,bottleneck_dropout))
        print('')

        return T, N, V, use_bottleneck, bottleneck_dropout



def get_scheduler(config, max_iterations ,optimizer):

    if config['optimisation']['use_scheduler']:

        print('Using learning rate scheduler')

        if config['optimisation']['scheduler'] == 'StepLR':

            scheduler = StepLR(optimizer=optimizer,
                                step_size= config['StepLR']['stepsize'],
                                gamma= config['StepLR']['decay'])
        
        elif config['optimisation']['scheduler'] == 'CosineDecay':

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                    T_max = config['CosineDecay']['T_max'],
                                                                    eta_min= config['CosineDecay']['eta_min'],
                                                                    )

        elif config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer,
                                            mode='max',
                                            factor=0.5,
                                            patience=2,
                                            cooldown=0,
                                            min_lr=0.0000001
                                                )

        if config['optimisation']['warmup']:

            scheduler = GradualWarmupScheduler(optimizer,
                                                multiplier=1, 
                                                total_epoch=config['optimisation']['nbr_step_warmup'], 
                                                after_scheduler=scheduler)
     
    else:
        # to use warmup without fancy scheduler
        if config['optimisation']['warmup']:
            scheduler = StepLR(optimizer,
                                step_size=max_iterations)

            scheduler = GradualWarmupScheduler(optimizer,
                                                multiplier=1, 
                                                total_epoch=config['optimisation']['nbr_step_warmup'], 
                                                after_scheduler=scheduler)
        else:

            return None
            
    return scheduler