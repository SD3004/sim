import os


def tensorboard_log_train(config, writer, scheduler, optimizer, loss_item, iter_count):
    
    writer.add_scalar('loss/train_it', loss_item, iter_count)

    if config['optimisation']['use_scheduler']:
        if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
            if config['optimisation']['warmup']:
                if (iter_count)<config['optimisation']['nbr_step_warmup']:
                    writer.add_scalar('LR',scheduler.get_lr()[0], iter_count)
                else:
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count)
            else:
                writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count)
        else:
            scheduler.step()
            writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count )
    else:
        if config['optimisation']['warmup']:
            scheduler.step()
            writer.add_scalar('LR',optimizer.param_groups[0]['lr'],iter_count )
        else:
            writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count )
    
    return scheduler, writer

def tensorboard_log_train_valset(writer, loss_item, iter_count):

    writer.add_scalar('loss/val', loss_item, iter_count)
    
    return writer


def tensorboard_log_train_all_clip_loss(config,
                                        writer,
                                        scheduler,
                                        optimizer,
                                        loss_fmri_video,
                                        loss_fmri_audio,
                                        loss_video_audio,
                                        iter_count):
    
    writer.add_scalar('loss/train_it_fmri_video', loss_fmri_video, iter_count)
    writer.add_scalar('loss/train_it_fmri_audio', loss_fmri_audio, iter_count)
    writer.add_scalar('loss/train_it_video_audio', loss_video_audio, iter_count)

    if config['optimisation']['use_scheduler']:
        if config['optimisation']['scheduler'] == 'ReduceLROnPlateau':
            if config['optimisation']['warmup']:
                if (iter_count)<config['optimisation']['nbr_step_warmup']:
                    writer.add_scalar('LR',scheduler.get_lr()[0], iter_count)
                else:
                    writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count)
            else:
                writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count)
        else:
            scheduler.step()
            writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count )
    else:
        if config['optimisation']['warmup']:
            scheduler.step()
            writer.add_scalar('LR',optimizer.param_groups[0]['lr'],iter_count )
        else:
            writer.add_scalar('LR',optimizer.param_groups[0]['lr'], iter_count )
    
    return scheduler, writer