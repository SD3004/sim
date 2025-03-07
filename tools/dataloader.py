import sys
import torch

sys.path.append('./')
sys.path.append('./tools')

from tools.datasets import dataset_clip_video_audio_fmri

from torch.utils.data.distributed import DistributedSampler


##### METRICS DATALOADER ########

def loader_tfmri_clip(data_path,
                config,
                world_size=1,
                rank=0):
    
    bs_train = config['training']['bs']
    bs_val = config['training']['bs_val']

    ###############################################################
    #####################    TRAINING DATA    #####################
    ###############################################################

    train_dataset = dataset_clip_video_audio_fmri(config=config,
                                                    data_path=data_path,
                                                    split='train',
                                                    )
    
    if config['distributed_training']:
        print('Setting up DistributedSampler: {}, {}'.format(world_size, rank))
        sampler_distributed = DistributedSampler(
                                        train_dataset,
                                        num_replicas = world_size,
                                        rank=rank,
                                            )
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = bs_train,
                                                        shuffle=False,
                                                        sampler=sampler_distributed,
                                                        num_workers=config['training']['num_workers'],
                                                        pin_memory=True)

    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size = bs_train,
                                                        shuffle=(not config['RECONSTRUCTION']),
                                                        num_workers=config['training']['num_workers'],
                                                        pin_memory=True)
        

    ###############################################################
    ####################    VALIDATION DATA    ####################
    ###############################################################

    #if cross validation then test set = validation set
    
    val_dataset = dataset_clip_video_audio_fmri(data_path=data_path,
                                                config=config,
                                                split='val')
    if config['distributed_training']:
        val_sampler_distributed = DistributedSampler(
                                        val_dataset,
                                        num_replicas = world_size,
                                        rank=rank,
                                            )
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                        batch_size = bs_val,
                                                        shuffle=False,
                                                        sampler=val_sampler_distributed,
                                                        num_workers=config['training']['num_workers'])
        
    else:
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size=bs_val,
                                                shuffle=False,
                                                num_workers=config['training']['num_workers'])
                    
    print('')
    print('#'*30)
    print('############ Data ############')
    print('#'*30)
    print('')

    print('')
    print('Training data: {}'.format(len(train_dataset)))
    print('Validation data: {}'.format(len(val_dataset)))
    
    if config['distributed_training']:
        return train_loader, val_loader, sampler_distributed, val_sampler_distributed
    else:
        return train_loader, val_loader
