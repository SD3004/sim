# -*- coding: utf-8 -*-
# @Author: Simon Dahan
#
# Created on Fri Oct 01 2021
#
# by Simon Dahan @SD3004
#
# Copyright (c) 2025 MeTrICS Lab
#torchrun --nproc_per_node=1 --nnodes=1  train_fmri_clip_ddp.py ../config/CLIP-video/hparams.yml

import os
import argparse
import yaml
import sys
import time
import glob
#remove warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from tqdm import tqdm
sys.path.append('../')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.sit import SiT
from models.clip import CLIPfMRIVideo, CLIPfMRIAudio, CLIPfMRIVideoAudio

from tools.utils import logging_clip, get_data_path, get_dataloaders_clip_distributed, get_dimensions, get_scheduler

from tools.log import tensorboard_log_train,tensorboard_log_train_all_clip_loss,tensorboard_log_train_valset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def cleanup():
    dist.destroy_process_group()

def train(config):

    ###########################################
    ######     DISTRIBUTED TRAINING      ######
    ###########################################

    ## distributed training
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)
    if rank ==0:
        print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")
        print(rank, device, torch.cuda.device_count())
        
    concat_batches = config['clip']['concat_batches']
    assert config['training']['bs'] == config['training']['bs_val']

    #############################
    ######     CONFIG      ######
    #############################

    if rank ==0:
        
        print('')
        print('#'*30)
        print('########### Config ###########')
        print('#'*30)
        print('')

    #mesh_resolution
    ico_mesh = config['mesh_resolution']['ico_mesh']
    ico_grid = config['mesh_resolution']['ico_grid']
    num_patches = config['ico_{}_grid'.format(ico_grid)]['num_patches']
    num_vertices = config['ico_{}_grid'.format(ico_grid)]['num_vertices']

    #data
    dataset = config['data']['dataset']
    #configuration = config['data']['configuration'] ## TO REMOVE
    #hemi = config['data']['hemi'] ## TO REMOVE

    #training
    gpu = config['training']['gpu']
    LR = config['training']['LR']
    use_confounds = config['training']['use_confounds'] ## TO REMOVE
    early_stopping = config['training']['early_stopping']
    dataloader = config['data']['dataloader']

    #if hemi == 'full':  ### TO IMPLEMENT ## TO REMOVE
    #    num_patches*=2
    try:
        print('creating a new data path')
        data_path = get_data_path(config)
    except:
        raise("can't get data path")

    #device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
    
    assert config['transformer']['dim'] == config['clip']['fmri_embedding_dims']

    if rank ==0:
        print('gpu: {}'.format(device))   
        print('dataset: {}'.format(dataset))  
        print('use confounds: {}'.format(use_confounds)) ## TO REMOVE
        print('model: {}'.format(config['MODEL']))
        #print('configuration: {}'.format(configuration))  
        print('data path: {}'.format(data_path))

    ##############################
    ######     DATASET      ######
    ##############################
    if rank ==0:
        print('')
        print('Mesh resolution - ico {}'.format(ico_mesh))
        print('Grid resolution - ico {}'.format(ico_grid))
        print('Number of patches - {}'.format(num_patches))
        print('Number of vertices - {}'.format(num_vertices))
        #print('Reorder patches: {}'.format(config['mesh_resolution']['reorder'])) ## TO REMOVE
        print('')

    try:
        if str(dataloader)=='bold':
            train_loader, val_loader,  train_sampler, val_sampler = get_dataloaders_clip_distributed(config,
                                                                                                     data_path,
                                                                                                     world_size,
                                                                                                     rank)
    except:
        raise("can't get dataloaders")
    
    ##############################
    ######      LOGGING     ######
    ##############################

    if rank==0:

        print('')
        print('#'*30)
        print('########## Logging ###########')
        print('#'*30)
        print('')

    # creating folders for logging. 
    if config['MODEL'] == 'clip-video' or config['MODEL'] == 'clip-audio' or config['MODEL'] == 'clip-fmri-video-audio':
        folder_to_save_model = logging_clip(config)
    else:
        raise('not implemented yet')
    
    if rank==0:
        try:
            os.makedirs(folder_to_save_model,exist_ok=False)
            print('Creating folder: {}'.format(folder_to_save_model))
        except OSError:
            print('folder already exist: {}'.format(folder_to_save_model))

    #Continue training. Training has been killed ? 
    pt_files = glob.glob(os.path.join(folder_to_save_model, '*.pt'))

    # Check if the list of files is not empty
    if pt_files:
        if rank == 0:
            print("There are .pt files in the folder.")
            for file in pt_files:
                print(file)
        continue_training = True

        ## log restart config file:
        config['RESTART_TRAINING_ID'] = int(config['RESTART_TRAINING_ID']) + 1
        with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)
    else:
        if rank == 0:
            print("There are no .pt files in the folder.")
        continue_training = False

    
    #tensorboard
    if rank==0:
        writer = SummaryWriter(log_dir=folder_to_save_model)

    ##############################
    #######     MODEL      #######
    ##############################

    if rank ==0:

        print('')
        print('#'*35)
        print('######### Init fMRI model #########')
        print('#'*35)
        print('')

    if config['MODEL'] == 'clip-video' or config['MODEL'] == 'clip-audio' or config['MODEL'] == 'clip-fmri-video-audio':    

        T, N, V, use_bottleneck, bottleneck_dropout = get_dimensions(config)

    if config['clip']['fmri_encoder'] == 'sit':
        fmri_model = SiT(dim=config['transformer']['dim'],
                        depth=config['transformer']['depth'],
                        heads=config['transformer']['heads'],
                        pool=config['transformer']['pool'], 
                        num_patches=N,
                        num_classes=config['transformer']['num_classes'],
                        num_channels=T,
                        num_vertices=V,
                        dim_head=config['transformer']['dim_head'],
                        dropout=config['transformer']['dropout'],
                        emb_dropout=config['transformer']['emb_dropout'],
                        use_pe=config['transformer']['use_pos_embedding'],
                        bottleneck_dropout=bottleneck_dropout,
                        use_bottleneck=use_bottleneck,
                        use_confounds=use_confounds,
                        weights_layers_init=config['transformer']['weights_layers_init'],
                        use_class_token=config['transformer']['use_class_token'],
                        trainable_pos_emb=config['transformer']['trainable_pos_emb'],
                        no_class_token_emb = config['transformer']['no_class_token_emb'],)
    elif config['clip']['fmri_encoder'] == 'mlp':
        fmri_model = SimpleMLP(input_size=T*N*V, 
                            output_size=config['clip']['clip_embedding_dimension'])

    ###################################
    #######     CLIP MODEL      #######
    ###################################
    
    if rank ==0:

        print('')
        print('#'*35)
        print('######### Init CLIP model #########')
        print('#'*35)
        print('')
        
    if config['MODEL'] == 'clip-video':

        model = CLIPfMRIVideo(video_model=config['video_model']['video_name'],
                                fmri_encoder=fmri_model,
                                fmri_encoder_model=config['clip']['fmri_encoder'],
                                video_encoder_trainable=config['clip']['video_encoder_trainable'],
                                fmri_encoder_trainable=config['clip']['fmri_encoder_trainable'],
                                use_class_token_fmri_encoder=config['clip']['use_class_token_fmri_encoder'],
                                clip_embedding_dimension=config['clip']['clip_embedding_dimension'],
                                fmri_embedding_dims=config['clip']['fmri_embedding_dims'],
                                video_embedding_dims=config['clip']['video_embedding_dims'],
                                dropout=config['clip']['dropout'],
                                temperature=config['clip']['temperature'],
                                video_loader = config['video_model']['video_loader'],
                                avg=config['clip']['avg_fmri'],
                                world_size=world_size,
                                device= device,
                                clip_loss_version = config['clip']['clip_loss_version']
                                )
        
        if rank==0:
            
            print('')   
            print('#'*60)
            print('Number of parameters all model: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
            print('Number of parameters fmri encoder model: {:,}'.format(sum(p.numel() for p in model.fmri_encoder.parameters() if p.requires_grad)))
            if config['video_model']['video_loader'] == 'video_encoder':
                print('Number of parameters video encoder model: {:,}'.format(sum(p.numel() for p in model.video_encoder.parameters() if p.requires_grad)))
            print('Number of parameters fmri wrapper encoder model: {:,}'.format(sum(p.numel() for p in model.fmri_wrapper.parameters() if p.requires_grad)))
            print('Number of parameters video wrapper model: {:,}'.format(sum(p.numel() for p in model.video_wrapper.parameters() if p.requires_grad)))
            print('#'*60)
            print('')
    
    elif config['MODEL'] == 'clip-audio':
        
        model = CLIPfMRIAudio(audio_model=config['audio_model']['audio_name'],
                                fmri_encoder=fmri_model,
                                fmri_encoder_model=config['clip']['fmri_encoder'],
                                audio_encoder_trainable=config['clip']['audio_encoder_trainable'],
                                fmri_encoder_trainable=config['clip']['fmri_encoder_trainable'],
                                use_class_token_fmri_encoder=config['clip']['use_class_token_fmri_encoder'],
                                clip_embedding_dimension=config['clip']['clip_embedding_dimension'],
                                fmri_embedding_dims=config['clip']['fmri_embedding_dims'],
                                audio_embedding_dims=config['clip']['audio_embedding_dims'],
                                dropout=config['clip']['dropout'],
                                temperature=config['clip']['temperature'],
                                audio_loader = config['audio_model']['audio_loader'],
                                avg=config['clip']['avg_fmri'],
                                world_size=world_size,
                                clip_loss_version = config['clip']['clip_loss_version'],
                                device= device
                                )
        if rank==0:
            
            print('')   
            print('#'*60)
            print('Number of parameters all model: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
            print('Number of parameters fmri encoder model: {:,}'.format(sum(p.numel() for p in model.fmri_encoder.parameters() if p.requires_grad)))
            if config['audio_model']['audio_loader'] == 'audio_encoder':
                print('Number of parameters audio encoder model: {:,}'.format(sum(p.numel() for p in model.audio_encoder.parameters() if p.requires_grad)))
            print('Number of parameters fmri wrapper encoder model: {:,}'.format(sum(p.numel() for p in model.fmri_wrapper.parameters() if p.requires_grad)))
            print('Number of parameters audio wrapper model: {:,}'.format(sum(p.numel() for p in model.audio_wrapper.parameters() if p.requires_grad)))
            print('#'*60)
            print('')
        
    elif config['MODEL'] == 'clip-fmri-video-audio':

        model = CLIPfMRIVideoAudio(video_model=config['video_model']['video_name'],
                                   audio_model=config['audio_model']['audio_name'],
                                    fmri_encoder=fmri_model,
                                    fmri_encoder_model=config['clip']['fmri_encoder'],
                                    video_encoder_trainable=config['clip']['video_encoder_trainable'],
                                    audio_encoder_trainable=config['clip']['audio_encoder_trainable'],
                                    fmri_encoder_trainable=config['clip']['fmri_encoder_trainable'],
                                    use_class_token_fmri_encoder=config['clip']['use_class_token_fmri_encoder'],
                                    clip_embedding_dimension=config['clip']['clip_embedding_dimension'],
                                    fmri_embedding_dims=config['clip']['fmri_embedding_dims'],
                                    video_embedding_dims=config['clip']['video_embedding_dims'],
                                    audio_embedding_dims=config['clip']['audio_embedding_dims'],
                                    dropout=config['clip']['dropout'],
                                    temperature_video=config['clip']['temperature_video'],
                                    temperature_audio=config['clip']['temperature_audio'],
                                    video_loader = config['video_model']['video_loader'],
                                    audio_loader = config['audio_model']['audio_loader'],
                                    avg=config['clip']['avg_fmri'],
                                    world_size=world_size,
                                    device= device,
                                    clip_loss_version = config['clip']['clip_loss_version']
                                    )
        
        if rank==0:
            
            print('')   
            print('#'*60)
            print('Number of parameters all model: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
            print('Number of parameters fmri encoder model: {:,}'.format(sum(p.numel() for p in model.fmri_encoder.parameters() if p.requires_grad)))
            if config['video_model']['video_loader'] == 'video_encoder':
                print('Number of parameters video encoder model: {:,}'.format(sum(p.numel() for p in model.video_encoder.parameters() if p.requires_grad)))
            if config['audio_model']['audio_loader'] == 'audio_encoder':
                print('Number of parameters audio encoder model: {:,}'.format(sum(p.numel() for p in model.audio_encoder.parameters() if p.requires_grad)))
            print('Number of parameters fmri wrapper encoder model: {:,}'.format(sum(p.numel() for p in model.fmri_wrapper.parameters() if p.requires_grad)))
            print('Number of parameters video wrapper model: {:,}'.format(sum(p.numel() for p in model.video_wrapper.parameters() if p.requires_grad)))
            print('Number of parameters audio wrapper model: {:,}'.format(sum(p.numel() for p in model.audio_wrapper.parameters() if p.requires_grad)))
            print('#'*60)
            print('')
            print('weights: {}'.format(config['clip']['weights']))
    

    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)

    if continue_training:

        ## if continue_traiing, need to reload the weights for the best encoder and wrappers

        if config['training']['finetuning_fmri_model']:
            # in that case there will be an encoder ckpt in the folder
            checkpoint = torch.load(os.path.join(folder_to_save_model,'fmri-encoder-best.pt'), map_location=lambda storage, loc: storage.cuda(rank))
            model.module.fmri_encoder.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            if rank==0:
                print('##### Loaded BEST checkpoint for fMRI encoder model successfully #####')
        else:
            raise NotImplementedError

        checkpoint = torch.load(os.path.join(folder_to_save_model,'fmri-wrapper-best.pt'), map_location=lambda storage, loc: storage.cuda(rank))
        model.module.fmri_wrapper.load_state_dict(checkpoint['model_state_dict'],strict=True) 
        if rank==0:
            print('##### Loading BEST checkpoint for fMRI wrapper model successfully #####')

        if config['MODEL'] == 'clip-video' or config['MODEL'] == 'clip-fmri-video-audio':
            
            checkpoint = torch.load(os.path.join(folder_to_save_model,'video-wrapper-best.pt'), map_location=lambda storage, loc: storage.cuda(rank))
            model.module.video_wrapper.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            if rank==0:
                print('##### Loading BEST checkpoint for video wrapper model successfully #####')
            
        if config['MODEL'] == 'clip-audio' or config['MODEL'] == 'clip-fmri-video-audio':

            checkpoint = torch.load(os.path.join(folder_to_save_model,'audio-wrapper-best.pt'), map_location=lambda storage, loc: storage.cuda(rank))
            model.module.audio_wrapper.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            if rank==0:
                print('##### Loading BEST checkpoint for audio wrapper model successfully #####')

    ### Loading weights for fMRI ENCODER to start the training ###
    else:
        if rank==0:
            print('*** Starting training from the beginning ***')

        if config['transformer']['init_weights']=='ssl_svmae':

            checkpoint = torch.load(config['weights']['ssl_svmae'])
            strict = config['training']['strict_weights']
            if rank == 0:
                print('')
                print('#'*15)
                print('Loading svMAE weights into transformer encoder')
                print('strict: {}'.format(strict))

            #import pdb;pdb.set_trace()
            model_state_dict = model.module.fmri_encoder.encoder.state_dict()
            # Update only the matching keys and shapes
            for name, param in checkpoint['model_state_dict'].items():
                if name in model_state_dict:
                    if param.size() == model_state_dict[name].size():
                        if rank==0:
                            print('{} is in the new dictionary and loaded correctly'.format(name))
                        model_state_dict[name].copy_(param)
                    else:
                        if rank == 0:
                            print('***')
                            print(f"Skipping {name} due to size mismatch: checkpoint {param.size()}, model {model_state_dict[name].size()}")
                            print('***')
            
            #import pdb;pdb.set_trace()
            model.module.fmri_encoder.encoder.load_state_dict(model_state_dict,strict=True)
            if rank == 0:
                print('#'*15)
                print('')
            #import pdb;pdb.set_trace()
        else:
            if rank == 0:
                print('Training sit encoder model from scratch')
  
    ## SET TRAINABLE PARAMETERS

    for p in model.module.fmri_encoder.parameters():
        p.requires_grad = config['clip']['fmri_encoder_trainable']
    
    for p in model.module.fmri_wrapper.parameters():
            p.requires_grad = True

    #video wrapper
    if config['MODEL'] == 'clip-video' or config['MODEL'] == 'clip-fmri-video-audio':
        for p in model.module.video_wrapper.parameters():
            p.requires_grad = True
    #audio wrapper
    if config['MODEL'] == 'clip-audio' or config['MODEL'] == 'clip-fmri-video-audio':
        for p in model.module.audio_wrapper.parameters():
            p.requires_grad = True

    if rank==0:
        print('')   
        print('#'*60)
        print('Number of trainable parameters all model: {:,}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
        print('Number of trainable parameters fmri encoder model: {:,}'.format(sum(p.numel() for p in model.module.fmri_encoder.parameters() if p.requires_grad)))
        if config['MODEL'] == 'clip-video':
            if config['video_model']['video_loader'] == 'video_encoder':
                print('Number of trainable parameters video encoder model: {:,}'.format(sum(p.numel() for p in model.module.video_encoder.parameters() if p.requires_grad)))
        if config['MODEL'] == 'clip-audio':
            if config['audio_model']['audio_loader'] == 'audio_encoder':
                print('Number of trainable parameters audio encoder model: {:,}'.format(sum(p.numel() for p in model.module.audio_encoder.parameters() if p.requires_grad)))
        print('Number of trainable parameters fmri wrapper encoder model: {:,}'.format(sum(p.numel() for p in model.module.fmri_wrapper.parameters() if p.requires_grad)))
        
        if config['MODEL'] == 'clip-video':
            print('Number of trainable parameters video wrapper model: {:,}'.format(sum(p.numel() for p in model.module.video_wrapper.parameters() if p.requires_grad)))
        elif config['MODEL'] == 'clip-audio':
            print('Number of trainable parameters audio wrapper model: {:,}'.format(sum(p.numel() for p in model.module.audio_wrapper.parameters() if p.requires_grad)))
        elif config['MODEL'] == 'clip-fmri-video-audio':
            print('Number of trainable parameters video wrapper model: {:,}'.format(sum(p.numel() for p in model.module.video_wrapper.parameters() if p.requires_grad)))
            print('Number of trainable parameters audio wrapper model: {:,}'.format(sum(p.numel() for p in model.module.audio_wrapper.parameters() if p.requires_grad)))
        print('#'*60)
        print('')
      
    #####################################
    #######     OPTIMISATION      #######
    #####################################

    if config['optimisation']['optimiser']=='SGD':
        if rank==0:
            print('Using SGD optimiser')
        optimizer = optim.SGD(model.parameters(), lr=LR, 
                                                weight_decay=config['SGD']['weight_decay'],
                                                momentum=config['SGD']['momentum'],
                                                nesterov=config['SGD']['nesterov'])
                                                
    elif config['optimisation']['optimiser']=='Adam':
        if rank==0:
            print('Using Adam optimiser')
        optimizer = optim.Adam(model.parameters(), lr=LR,
                                weight_decay=config['Adam']['weight_decay'])
      
    elif config['optimisation']['optimiser']=='AdamW':
        if rank==0:
            print('Using AdamW optimiser')
        optimizer = optim.AdamW(model.parameters(),
                                lr=LR,
                                weight_decay=config['AdamW']['weight_decay'])
    else:
        raise('not implemented yet')

    if continue_training:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        iter_count = checkpoint['iteration']
        running_loss = checkpoint['loss']
        best_loss = running_loss
        if rank==0:
            print('')
            print('#### LOADING OPTIMIZER STATE ####')
            print('Loading successfully')
            print('Starting training from iteration  {}'.format(iter_count))
        
    else:
        iter_count= 0
        running_loss = 0 
        best_loss = 1000000000000000
        if rank==0:
            print('Training from scratch')

    ###################################
    #######     SCHEDULING      #######
    ###################################
    
    max_iterations = config['training']['iterations']
    log_val_it = config['training']['log_val_it']
    scheduler = get_scheduler(config, max_iterations-iter_count, optimizer)

    ##############################
    ######     TRAINING     ######
    ##############################
    
    if rank==0:

        print('')
        print('#'*30)
        print('###### Starting training #####')
        print('#'*30)
        print('')

     # classification and regression tasks
    c_early_stop = 0     
    t0 = time.time()
    
    if rank==0:
        tqdm_object = tqdm(total=max_iterations-iter_count,)
    print('')
    epoch = 0
    #import pdb;pdb.set_trace()
    while iter_count < max_iterations:
        
        train_sampler.set_epoch(epoch)
            
        for i, data in enumerate(train_loader):
                        
            model.train()

            ##### FORWARD & BACKWARD PASS #####
            if config['MODEL']=='clip-video':
                inputs_fmri, stimulus_input = data[0].to(device), data[1]
            elif config['MODEL']=='clip-audio':
                inputs_fmri, stimulus_input = data[0].to(device), data[1]
            elif config['MODEL']=='clip-fmri-video-audio':
                inputs_fmri, stimulus_video_input, stimulus_audio_input = data[0].to(device), data[1].to(device), data[2].to(device)

            if config['training']['runtime']:
                B,b,t,n,v = inputs_fmri.shape
            else:
                B,t,n,v = inputs_fmri.shape
            #if config['fMRI']['nbr_clip_sampled_fmri']==1:
            #    inputs_fmri = inputs_fmri.squeeze(1)
            #else:
            inputs_fmri = inputs_fmri.view(-1,t,n,v)
            if config['MODEL'] == 'clip-audio' and config['audio_model']['audio_loader']=='audio_encoder':
                stimulus_input = stimulus_input.view(-1,stimulus_input.shape[-1])
                
                    
            #print(inputs_fmri.shape)
            if config['MODEL'] == 'clip-fmri-video-audio':
                w = config['clip']['weights']
                loss_fmri_video, loss_fmri_audio, loss_video_audio = model(inputs_fmri, stimulus_video_input,stimulus_audio_input, concat_batches, dist) 
                clip_loss = w[0]*loss_fmri_video + w[1]*loss_fmri_audio + w[2]*loss_video_audio
                clip_loss /= 6 
                optimizer.zero_grad()
                clip_loss.backward()
                optimizer.step()
                if rank==0:
                    tqdm_object.update(1)
                    tqdm_object.set_postfix(train_loss=clip_loss, train_time=round(time.time()-t0,4))
                    scheduler, writer = tensorboard_log_train_all_clip_loss(config, 
                                                                            writer, 
                                                                            scheduler, 
                                                                            optimizer, 
                                                                            loss_fmri_video,
                                                                            loss_fmri_audio,
                                                                            loss_video_audio,
                                                                            iter_count+1)
            else:
                clip_loss = model(inputs_fmri, stimulus_input, concat_batches, dist)  
                optimizer.zero_grad()
                clip_loss.backward()
                optimizer.step()
                if rank==0:
                    tqdm_object.update(1)
                    tqdm_object.set_postfix(train_loss=clip_loss, train_time=round(time.time()-t0,4))
                    scheduler, writer = tensorboard_log_train(config, writer, scheduler, optimizer, clip_loss, iter_count+1)
                        
        
            ##############################
            ######    VALIDATION    ######
            ##############################

            if (iter_count+1)%log_val_it==0:

                # Ensure all processes have finished their part of training before starting validation
                torch.distributed.barrier()

                running_val_loss = 0
                model.eval()
                val_sampler.set_epoch(epoch)

                with torch.no_grad():

                    for i, data in enumerate(val_loader):
                        
                        ##### FORWARD & BACKWARD PASS #####
                        if config['MODEL']=='clip-video':
                            inputs_fmri, stimulus_input = data[0].to(device), data[1]
                        elif config['MODEL']=='clip-audio':
                            inputs_fmri, stimulus_input = data[0].to(device), data[1]
                        elif config['MODEL']=='clip-fmri-video-audio':
                            inputs_fmri, stimulus_video_input, stimulus_audio_input = data[0].to(device), data[1].to(device), data[2].to(device)

                        if config['training']['runtime']:
                            B,b,t,n,v = inputs_fmri.shape
                        else:
                            B,t,n,v = inputs_fmri.shape
                        #if config['fMRI']['nbr_clip_sampled_fmri']==1:
                        #    inputs_fmri = inputs_fmri.squeeze(1)
                        #else:
                        inputs_fmri = inputs_fmri.view(-1,t,n,v)
                        if config['MODEL'] == 'clip-audio' and config['audio_model']['audio_loader']=='audio_encoder':
                            stimulus_input = stimulus_input.view(-1,stimulus_input.shape[-1])
                            
                        if config['MODEL'] == 'clip-fmri-video-audio':
                            loss_fmri_video, loss_fmri_audio, loss_video_audio = model(inputs_fmri, stimulus_video_input,stimulus_audio_input, concat_batches, dist) 
                            clip_loss = loss_fmri_video + loss_fmri_audio + loss_video_audio
                            clip_loss /= 3 
                        else:
                            clip_loss = model(inputs_fmri, stimulus_input, concat_batches, dist) 
                        
                        running_val_loss += clip_loss.item()
                                        
                loss_val_epoch = running_val_loss/(i+1)

                if rank==0:

                    writer = tensorboard_log_train_valset(writer, loss_val_epoch,iter_count+1)
                    
                    print('| Validation | Iteration - {} | Loss - {} | Time - {}'.format(iter_count+1, round(loss_val_epoch,4), round(time.time()-t0,4)))
                    
                    if (loss_val_epoch < best_loss) and (config['training']['save_ckpt']):
                            
                        config['results'] = {}
                        config['results']['best_iteration'] = iter_count+1
                        config['results']['last_loss'] = clip_loss.item()
                        config['results']['best_current_loss_validation'] = loss_val_epoch

                        with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                                yaml.dump(config, yaml_file)
                                                                
                        torch.save({ 'iteration':iter_count+1,
                                    'model_state_dict': model.module.fmri_encoder.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss_validation':loss_val_epoch,
                                    'loss': clip_loss.item()
                                    },
                                    os.path.join(folder_to_save_model, 'fmri-encoder-best.pt'))
                        
                        torch.save({ 'iteration':iter_count+1,
                                    'model_state_dict': model.module.fmri_wrapper.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss_validation':loss_val_epoch,
                                    'loss': clip_loss.item()
                                    },
                                    os.path.join(folder_to_save_model, 'fmri-wrapper-best.pt'))
                        
                        if config['MODEL'] == 'clip-video':
                            torch.save({ 'iteration':iter_count+1,
                                        'model_state_dict': model.module.video_wrapper.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'loss_validation':loss_val_epoch,
                                        'loss': clip_loss.item()
                                        },
                                        os.path.join(folder_to_save_model, 'video-wrapper-best.pt'))
                            
                        elif config['MODEL'] == 'clip-audio':
                            torch.save({ 'iteration':iter_count+1,
                                        'model_state_dict': model.module.audio_wrapper.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'loss_validation':loss_val_epoch,
                                        'loss': clip_loss.item()
                                        },
                                        os.path.join(folder_to_save_model, 'audio-wrapper-best.pt'))
                        elif config['MODEL'] == 'clip-fmri-video-audio':
                            
                            torch.save({ 'iteration':iter_count+1,
                                        'model_state_dict': model.module.video_wrapper.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'loss_validation':loss_val_epoch,
                                        'loss': clip_loss.item()
                                        },
                                        os.path.join(folder_to_save_model, 'video-wrapper-best.pt'))
                            
                            
                            torch.save({ 'iteration':iter_count+1,
                                        'model_state_dict': model.module.audio_wrapper.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'loss_validation':loss_val_epoch,
                                        'loss': clip_loss.item()
                                        },
                                        os.path.join(folder_to_save_model, 'audio-wrapper-best.pt'))

                        with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
                            yaml.dump(config, yaml_file)
            
                            
            iter_count+= 1
            
            if iter_count >= max_iterations:
                break
            
        epoch+=1
        if early_stopping and (c_early_stop>=early_stopping):
            print('stop training - early stopping')
            break
        
    if early_stopping and (c_early_stop>=early_stopping):
        config['results']['training_finished'] = 'early stopping' 
    else:
        config['results']['training_finished'] = True 

    #####################################
    ######    SAVING FINAL CKPT    ######
    #####################################
    
    #import pdb;pdb.set_trace()

    if config['training']['save_ckpt']:
        torch.save({ 'iteration':iter_count+1,
                    'model_state_dict': model.module.fmri_encoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_validation':loss_val_epoch,
                    'loss': clip_loss.item()
                    },
                    os.path.join(folder_to_save_model, 'fmri-encoder-final.pt'))
        
        torch.save({ 'iteration':iter_count+1,
                    'model_state_dict': model.module.fmri_wrapper.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_validation':loss_val_epoch,
                    'loss': clip_loss.item()
                    },
                    os.path.join(folder_to_save_model, 'fmri-wrapper-final.pt'))
        
        if config['MODEL'] == 'clip-video':
            torch.save({ 'iteration':iter_count+1,
                        'model_state_dict': model.module.video_wrapper.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_validation':loss_val_epoch,
                        'loss': clip_loss.item()
                        },
                        os.path.join(folder_to_save_model, 'video-wrapper-final.pt'))
        elif config['MODEL'] == 'clip-audio':
            torch.save({ 'iteration':iter_count+1,
                        'model_state_dict': model.module.audio_wrapper.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_validation':loss_val_epoch,
                        'loss': clip_loss.item()
                        },
                        os.path.join(folder_to_save_model, 'audio-wrapper-final.pt'))
            
        elif config['MODEL'] == 'clip-fmri-video-audio':
            torch.save({ 'iteration':iter_count+1,
                        'model_state_dict': model.module.video_wrapper.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_validation':loss_val_epoch,
                        'loss': clip_loss.item()
                        },
                        os.path.join(folder_to_save_model, 'video-wrapper-final.pt'))
            
            torch.save({ 'iteration':iter_count+1,
                        'model_state_dict': model.module.audio_wrapper.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss_validation':loss_val_epoch,
                        'loss': clip_loss.item()
                        },
                        os.path.join(folder_to_save_model, 'audio-wrapper-final.pt'))

        
        with open(os.path.join(folder_to_save_model,'hparams.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='ViT')

    parser.add_argument(
                        'config',
                        type=str,
                        default='./config/hparams.yml',
                        help='path where the data is stored')
    
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Call training
    train(config)
