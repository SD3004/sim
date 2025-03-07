from einops import repeat,rearrange
import torch
import torch.nn as nn
from torch import nn, einsum


import torch.nn.functional as F
import numpy as np

from models.wrapper import Wrapper 

from mmengine.dataset import Compose
#from mmaction2.mmaction.apis import inference_recognizer, init_recognizer

import torchaudio

from .gather import GatherLayer


###https://github.com/zsnoob/EfficientDDP-4-Contrastive-Train
##implementation from https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2


class CLIPfMRIVideo(nn.Module):

    def __init__(
        self,
        video_model,
        fmri_encoder, 
        fmri_encoder_model = 'sit',
        video_encoder_trainable = False, 
        fmri_encoder_trainable = False, 
        use_class_token_fmri_encoder = False,
        clip_embedding_dimension = 256,
        fmri_embedding_dims= 768,
        video_embedding_dims= 768,
        dropout=0.0,
        temperature = 1.0,
        video_loader = 'video_encoder',
        avg = 'prior_proj',
        device = 'cuda:0',
        world_size =  1,
        clip_loss_version = 'clip-v0'
    ):
        super().__init__()
        
        self.device = device
        self.video_loader = video_loader
        self.clip_loss_version = clip_loss_version
        self.use_class_token_fmri_encoder = use_class_token_fmri_encoder
        self.fmri_encoder_model = fmri_encoder_model
        print('video loader: {}'.format(video_loader))

        if self.fmri_encoder_model == 'sit':
            #set trainable parameters for video encoder:
            self.fmri_encoder = fMRIEncoder(encoder=fmri_encoder,
                                            trainable=fmri_encoder_trainable,
                                            use_class_token=use_class_token_fmri_encoder)
        else:
            raise NotImplementedError


        if video_loader == 'video_encoder':
            self.video_encoder = VideoEncoder(model_name=video_model, 
                                        trainable=video_encoder_trainable,
                                        device= device)
        
        #fmri wrapper
        self.fmri_wrapper = Wrapper(embedding_dim=fmri_embedding_dims,
                                    projection_dim=clip_embedding_dimension,
                                    dropout=dropout)
        
        #self.fmri_wrapper.to(device)
        
        #video wrapper
        self.video_wrapper = Wrapper(embedding_dim=video_embedding_dims,
                                    projection_dim=clip_embedding_dimension,
                                    dropout=dropout)
    
        #self.video_wrapper.to(device)
        
        self.temperature = temperature

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.avg = avg
        self.world_size = world_size
        

    def encoder_fmri(self, fmri):

        return self.fmri_encoder(fmri)

    def encoder_video(self,video):

        return self.video_encoder(video,self.device)
    
    def proj_fmri(self,fmri_features):

        return self.fmri_wrapper(fmri_features)
    
    def proj_video(self,video_features):

        return self.video_wrapper(video_features)

    def get_loss(self,fmri_embeddings, video_embeddings):

        # Calculating the Loss
        logits = (video_embeddings @ fmri_embeddings.T) / self.temperature
        fmri_similarity = fmri_embeddings @ fmri_embeddings.T
        video_similarity = video_embeddings @ video_embeddings.T
        targets = F.softmax(
            (fmri_similarity + video_similarity) / (2 * self.temperature), dim=-1
        )
        video_loss = cross_entropy(logits, targets, reduction='none')
        fmri_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (video_loss + fmri_loss) / 2.0 # shape: (batch_size)

        return loss.mean()
    

    def get_loss_openai(self,fmri_embeddings, video_embeddings):
        #https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
        #https://github.com/openai/CLIP/blob/main/clip/model.py
        #https://github.com/openai/CLIP/issues/83

        fmri_latents = F.normalize(fmri_embeddings, p=2, dim=-1)
        video_latents = F.normalize(video_embeddings, p=2, dim=-1)

        temp = nn.Parameter(torch.tensor(self.temperature))
        temp = temp.exp()

        sim = einsum('i d, j d -> i j', fmri_latents, video_latents) * temp
        labels = torch.arange(fmri_embeddings.shape[0], device = self.device)

        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        #import pdb;pdb.set_trace()
        return loss.mean()

    def get_loss_openai_ddp(self,fmri_embeddings, video_embeddings):
        #https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
        #https://github.com/openai/CLIP/blob/main/clip/model.py
        #https://github.com/openai/CLIP/issues/83
                
        
        fmri_embeddings = torch.cat(GatherLayer.apply(fmri_embeddings), dim=0)
        video_embeddings = torch.cat(GatherLayer.apply(video_embeddings), dim=0)

        fmri_latents = F.normalize(fmri_embeddings, p=2, dim=-1)
        video_latents = F.normalize(video_embeddings, p=2, dim=-1)

        temp = nn.Parameter(torch.tensor(self.temperature))
        temp = temp.exp()

        sim = einsum('i d, j d -> i j', fmri_latents, video_latents) * temp
        labels = torch.arange(fmri_embeddings.shape[0], device = self.device)

        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss.mean()
    
    def get_loss_ddp(self,fmri_embeddings, video_embeddings):
                
            
        fmri_embeddings = torch.cat(GatherLayer.apply(fmri_embeddings), dim=0)
        video_embeddings = torch.cat(GatherLayer.apply(video_embeddings), dim=0)
    
        # Calculating the Loss
        logits = (video_embeddings @ fmri_embeddings.T) / self.temperature
        fmri_similarity = fmri_embeddings @ fmri_embeddings.T
        video_similarity = video_embeddings @ video_embeddings.T
        targets = F.softmax(
            (fmri_similarity + video_similarity) / 2 * self.temperature, dim=-1
        )
        video_loss = cross_entropy(logits, targets, reduction='none')
        fmri_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (video_loss + fmri_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def forward(self,fmri, video_clips, concat_batches = False ,dist_config=False):
        if self.fmri_encoder_model == 'sit':
            #extract fMRI embeddings
            fmri_features = self.encoder_fmri(fmri)

            if self.use_class_token_fmri_encoder:
                fmri_features = fmri_features[:,1:,:]
            
            if self.avg == 'prior_proj':
                fmri_features = fmri_features.mean(dim=1)
                fmri_embeddings = self.proj_fmri(fmri_features)
            elif self.avg == 'post_proj':
                fmri_embeddings = self.proj_fmri(fmri_features)
                fmri_embeddings = fmri_embeddings.mean(dim=1)
        
        else:
            raise NotImplementedError
        #extract video embeddings
        if self.video_loader=='video_encoder':
            video_features = self.encoder_video(video_clips)
        elif self.video_loader=='numpy':
            video_features = video_clips.to(self.device)

        #project video embeddings
        video_embeddings = self.proj_video(video_features)
        #normalise features
        if dist_config and concat_batches:
            if self.clip_loss_version == 'clip-v0':
                loss = self.get_loss_ddp(fmri_embeddings, video_embeddings)
            elif self.clip_loss_version == 'clip-openai':
                loss = self.get_loss_openai_ddp(fmri_embeddings, video_embeddings)

        elif dist_config:
            loss = self.get_loss(fmri_embeddings, video_embeddings)
            dist_config.all_reduce(loss, op=dist_config.ReduceOp.SUM)
            loss /= self.world_size
        else:
            if self.clip_loss_version == 'clip-v0':
                loss = self.get_loss(fmri_embeddings, video_embeddings)
            elif self.clip_loss_version == 'clip-openai':
                loss = self.get_loss_openai(fmri_embeddings, video_embeddings)

        return loss


class CLIPfMRIVideoAudio(nn.Module):

    def __init__(
        self,
        video_model,
        audio_model,
        fmri_encoder, 
        fmri_encoder_model = 'sit',
        video_encoder_trainable = False, 
        audio_encoder_trainable = False, 
        fmri_encoder_trainable = False, 
        use_class_token_fmri_encoder = False,
        clip_embedding_dimension = 256,
        fmri_embedding_dims= 768,
        video_embedding_dims= 768,
        audio_embedding_dims= 768,
        dropout=0.0,
        temperature_video = 1.0,
        temperature_audio = 0.5,
        video_loader = 'video_encoder',
        audio_loader = 'audio_encoder', 
        avg = 'prior_proj',
        device = 'cuda:0',
        clip_loss_version = 'clip-openai',
        world_size =  1,
    ):
        super().__init__()
        
        self.device = device
        self.video_loader = video_loader
        self.audio_loader = audio_loader
        self.use_class_token_fmri_encoder = use_class_token_fmri_encoder
        self.clip_loss_version = clip_loss_version
        self.fmri_encoder_model = fmri_encoder_model
        print('video loader: {}'.format(video_loader))
        print('audio loader: {}'.format(audio_loader))

        #set trainable parameters for video encoder:
        if self.fmri_encoder_model == 'sit':
            self.fmri_encoder = fMRIEncoder(encoder=fmri_encoder,
                                            trainable=fmri_encoder_trainable,
                                            use_class_token=use_class_token_fmri_encoder)
        elif self.fmri_encoder_model == 'mlp':
            self.fmri_encoder = fmri_encoder


        if video_loader == 'video_encoder':
            self.video_encoder = VideoEncoder(model_name=video_model, 
                                        trainable=video_encoder_trainable,
                                        device= device)
            
        if audio_loader == 'audio_encoder':
            self.audio_encoder = AudioEncoder(model_name=audio_model, 
                                        trainable=audio_encoder_trainable,
                                        device= device)
        
        #fmri wrapper
        self.fmri_wrapper = Wrapper(embedding_dim=fmri_embedding_dims,
                                    projection_dim=clip_embedding_dimension,
                                    dropout=dropout)
                
        #video wrapper
        self.video_wrapper = Wrapper(embedding_dim=video_embedding_dims,
                                    projection_dim=clip_embedding_dimension,
                                    dropout=dropout)
        
        #video wrapper
        self.audio_wrapper = Wrapper(embedding_dim=audio_embedding_dims,
                                    projection_dim=clip_embedding_dimension,
                                    dropout=dropout)
    
        
        self.temperature_video = temperature_video
        self.temperature_audio = temperature_audio

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.avg = avg
        self.world_size = world_size
        

    def encoder_fmri(self, fmri):

        return self.fmri_encoder(fmri)

    def encoder_video(self,video):

        return self.video_encoder(video,self.device)
    
    def encoder_audio(self,audio):

        return self.audio_encoder(audio,self.device)
    
    def proj_fmri(self,fmri_features):

        return self.fmri_wrapper(fmri_features)
    
    def proj_video(self,video_features):

        return self.video_wrapper(video_features)

    def proj_audio(self,audio_features):

        return self.audio_wrapper(audio_features)

    def get_loss(self,fmri_embeddings, video_embeddings, temperature):

        # Calculating the Loss
        logits = (video_embeddings @ fmri_embeddings.T) / temperature
        fmri_similarity = fmri_embeddings @ fmri_embeddings.T
        video_similarity = video_embeddings @ video_embeddings.T
        targets = F.softmax(
            (fmri_similarity + video_similarity) / 2 * temperature, dim=-1
        )
        video_loss = cross_entropy(logits, targets, reduction='none')
        fmri_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (video_loss + fmri_loss) / 2.0 # shape: (batch_size)

        return loss.mean()
    
    def get_loss_ddp(self,fmri_embeddings, video_embeddings, temperature):
        #https://github.com/moein-shariatnia/OpenAI-CLIP
                
        
        fmri_embeddings = torch.cat(GatherLayer.apply(fmri_embeddings), dim=0)
        video_embeddings = torch.cat(GatherLayer.apply(video_embeddings), dim=0)
    
        # Calculating the Loss
        logits = (video_embeddings @ fmri_embeddings.T) / temperature
        fmri_similarity = fmri_embeddings @ fmri_embeddings.T
        video_similarity = video_embeddings @ video_embeddings.T
        targets = F.softmax(
            (fmri_similarity + video_similarity) / 2 * temperature, dim=-1
        )
        video_loss = cross_entropy(logits, targets, reduction='none')
        fmri_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (video_loss + fmri_loss) / 2.0 # shape: (batch_size)
        return loss.mean()
    
    def get_loss_openai_ddp(self,fmri_embeddings, video_embeddings, temperature):
        #https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
        #https://github.com/openai/CLIP/blob/main/clip/model.py
        #https://github.com/openai/CLIP/issues/83
        #https://github.com/moein-shariatnia/OpenAI-CLIP
                
        
        fmri_embeddings = torch.cat(GatherLayer.apply(fmri_embeddings), dim=0)
        video_embeddings = torch.cat(GatherLayer.apply(video_embeddings), dim=0)

        fmri_latents = F.normalize(fmri_embeddings, p=2, dim=-1)
        video_latents = F.normalize(video_embeddings, p=2, dim=-1)

        temp = nn.Parameter(torch.tensor(temperature))
        temp = temp.exp()

        sim = einsum('i d, j d -> i j', fmri_latents, video_latents) * temp
        labels = torch.arange(fmri_embeddings.shape[0], device = self.device)

        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss.mean()

    def forward(self,fmri, video_clips, audio_clips, concat_batches = False ,dist_config=False):
        
        if self.fmri_encoder_model == 'sit':
            #extract fMRI embeddings
            fmri_features = self.encoder_fmri(fmri)

            if self.use_class_token_fmri_encoder:
                fmri_features = fmri_features[:,1:,:]
            if self.avg == 'prior_proj':
                fmri_features = fmri_features.mean(dim=1)
                fmri_embeddings = self.proj_fmri(fmri_features)
            elif self.avg == 'post_proj':
                fmri_embeddings = self.proj_fmri(fmri_features)
                fmri_embeddings = fmri_embeddings.mean(dim=1)
        else:
            raise NotImplementedError
        #extract video embeddings
        if self.video_loader=='video_encoder':
            video_features = self.encoder_video(video_clips)
        elif self.video_loader=='numpy':
            video_features = video_clips.to(self.device)
            
        #extract audio embeddings
        if self.audio_loader=='audio_encoder':
            audio_features = self.encoder_audio(audio_clips)
        elif self.audio_loader=='numpy':
            audio_features = audio_clips.to(self.device)

        #project video embeddings
        video_embeddings = self.proj_video(video_features)
        audio_embeddings = self.proj_audio(audio_features)
        #normalise features
        if dist_config and concat_batches:
            if self.clip_loss_version == 'clip-v0':
                loss_fmri_video = self.get_loss_ddp(fmri_embeddings, video_embeddings,self.temperature_video)
                loss_fmri_audio = self.get_loss_ddp(fmri_embeddings, audio_embeddings,self.temperature_audio)
                loss_video_audio = self.get_loss_ddp(video_embeddings, audio_embeddings,self.temperature_audio)
            elif self.clip_loss_version == 'clip-openai':
                loss_fmri_video = self.get_loss_openai_ddp(fmri_embeddings, video_embeddings,self.temperature_video)
                loss_fmri_audio = self.get_loss_openai_ddp(fmri_embeddings, audio_embeddings,self.temperature_audio)
                loss_video_audio = self.get_loss_openai_ddp(video_embeddings, audio_embeddings,self.temperature_audio)
            else:
                raise NotImplementedError    
        elif dist_config:
            loss_fmri_video = self.get_loss(fmri_embeddings, video_embeddings, self.temperature_video)
            dist_config.all_reduce(loss_fmri_video, op=dist_config.ReduceOp.SUM)
            loss_fmri_video /= self.world_size
            
            loss_fmri_audio = self.get_loss(fmri_embeddings, audio_embeddings, self.temperature_audio)
            dist_config.all_reduce(loss_fmri_audio, op=dist_config.ReduceOp.SUM)
            loss_fmri_audio /= self.world_size
            
            loss_video_audio = self.get_loss(video_embeddings, audio_embeddings, self.temperature_audio)
            dist_config.all_reduce(loss_video_audio, op=dist_config.ReduceOp.SUM)
            loss_video_audio /= self.world_size
        else:
            loss_fmri_video = self.get_loss(fmri_embeddings, video_embeddings, self.temperature_video)            
            loss_fmri_audio = self.get_loss(fmri_embeddings, audio_embeddings, self.temperature_audio)
            loss_video_audio = self.get_loss(video_embeddings, audio_embeddings, self.temperature_audio)
            
        return loss_fmri_video, loss_fmri_audio, loss_video_audio

class CLIPfMRIAudio(nn.Module):

    def __init__(
        self,
        audio_model,
        fmri_encoder, 
        fmri_encoder_model = 'sit', 
        audio_encoder_trainable = False, 
        fmri_encoder_trainable = False, 
        use_class_token_fmri_encoder = False,
        clip_embedding_dimension = 256,
        fmri_embedding_dims= 768,
        audio_embedding_dims= 768,
        dropout=0.0,
        temperature = 1.0,
        audio_loader = 'audio_encoder',
        avg = 'prior_proj',
        device = 'cuda:0',
        world_size =  1,
        clip_loss_version = 'clip-v0'

    ):
        super().__init__()
        
        self.device = device
        self.audio_loader = audio_loader
        self.clip_loss_version = clip_loss_version
        self.use_class_token_fmri_encoder = use_class_token_fmri_encoder
        self.fmri_encoder_model = fmri_encoder_model
        print('audio loader: {}'.format(audio_loader))
        
        #set trainable parameters for video encoder:
        if self.fmri_encoder_model == 'sit':
            #set trainable parameters for video encoder:
            self.fmri_encoder = fMRIEncoder(encoder=fmri_encoder,
                                            trainable=fmri_encoder_trainable,
                                            use_class_token=use_class_token_fmri_encoder)
        
        else:
            raise NotImplementedError
        if audio_loader == 'audio_encoder':
            self.audio_encoder = AudioEncoder(model_name=audio_model, 
                                        trainable=audio_encoder_trainable,
                                        device= device)
    
        #fmri wrapper
        self.fmri_wrapper = Wrapper(embedding_dim=fmri_embedding_dims,
                                    projection_dim=clip_embedding_dimension,
                                    dropout=dropout)
        
        
        #audio wrapper
        self.audio_wrapper = Wrapper(embedding_dim=audio_embedding_dims,
                                    projection_dim=clip_embedding_dimension,
                                    dropout=dropout)
    
        
        self.temperature = temperature

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.avg = avg
        self.world_size = world_size

    def encoder_fmri(self, fmri):

        return self.fmri_encoder(fmri)

    def encoder_audio(self,audio):

        return self.audio_encoder(audio,self.device)

    def proj_fmri(self,fmri_features):

        return self.fmri_wrapper(fmri_features)
    
    def proj_audio(self,audio_features):

        return self.audio_wrapper(audio_features)

    def get_loss(self,fmri_embeddings, audio_embeddings):

        # Calculating the Loss
        logits = (audio_embeddings @ fmri_embeddings.T) / self.temperature
        fmri_similarity = fmri_embeddings @ fmri_embeddings.T
        audio_similarity = audio_embeddings @ audio_embeddings.T
        targets = F.softmax(
            (fmri_similarity + audio_similarity) / 2 * self.temperature, dim=-1
        )
        audio_loss = cross_entropy(logits, targets, reduction='none')
        fmri_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (audio_loss + fmri_loss) / 2.0 # shape: (batch_size)

        return loss.mean()

    def get_loss_ddp(self,fmri_embeddings, audio_embeddings):
                
    
        fmri_embeddings = torch.cat(GatherLayer.apply(fmri_embeddings), dim=0)
        audio_embeddings = torch.cat(GatherLayer.apply(audio_embeddings), dim=0)
    
        # Calculating the Loss
        logits = (audio_embeddings @ fmri_embeddings.T) / self.temperature
        fmri_similarity = fmri_embeddings @ fmri_embeddings.T
        audio_similarity = audio_embeddings @ audio_embeddings.T
        targets = F.softmax(
            (fmri_similarity + audio_similarity) / 2 * self.temperature, dim=-1
        )
        audio_loss = cross_entropy(logits, targets, reduction='none')
        fmri_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (audio_loss + fmri_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

    def get_loss_openai(self,fmri_embeddings, video_embeddings):
        #https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
        #https://github.com/openai/CLIP/blob/main/clip/model.py
        #https://github.com/openai/CLIP/issues/83

        fmri_latents = F.normalize(fmri_embeddings, p=2, dim=-1)
        video_latents = F.normalize(video_embeddings, p=2, dim=-1)

        temp = nn.Parameter(torch.tensor(self.temperature))
        temp = temp.exp()

        sim = einsum('i d, j d -> i j', fmri_latents, video_latents) * temp
        labels = torch.arange(fmri_embeddings.shape[0], device = self.device)

        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        #import pdb;pdb.set_trace()
        return loss.mean()

    def get_loss_openai_ddp(self,fmri_embeddings, audio_embeddings):
        #https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
        #https://github.com/openai/CLIP/blob/main/clip/model.py
        #https://github.com/openai/CLIP/issues/83
        fmri_embeddings = torch.cat(GatherLayer.apply(fmri_embeddings), dim=0)
        audio_embeddings = torch.cat(GatherLayer.apply(audio_embeddings), dim=0)

        fmri_latents = F.normalize(fmri_embeddings, p=2, dim=-1)
        audio_latents = F.normalize(audio_embeddings, p=2, dim=-1)

        temp = nn.Parameter(torch.tensor(self.temperature))
        temp = temp.exp()

        sim = einsum('i d, j d -> i j', fmri_latents, audio_latents) * temp
        labels = torch.arange(fmri_embeddings.shape[0], device = self.device)

        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss.mean()

    def forward(self,fmri, audio_clips, concat_batches = False ,dist_config=False):
        
        if self.fmri_encoder_model == 'sit':

            #extract fMRI embeddings
            fmri_features = self.encoder_fmri(fmri)

            if self.use_class_token_fmri_encoder:
                fmri_features = fmri_features[:,1:,:]
            if self.avg == 'prior_proj':
                fmri_features = fmri_features.mean(dim=1)
                fmri_embeddings = self.proj_fmri(fmri_features)
            elif self.avg == 'post_proj':
                fmri_embeddings = self.proj_fmri(fmri_features)
                fmri_embeddings = fmri_embeddings.mean(dim=1)
        else:
            raise NotImplementedError
        #extract video embeddings
        if self.audio_loader=='audio_encoder':
            audio_features = self.encoder_audio(audio_clips)
        elif self.audio_loader=='numpy':
            audio_features = audio_clips.to(self.device)
            
            

         #project video embeddings
        audio_embeddings = self.proj_audio(audio_features)
        #normalise features
        if dist_config and concat_batches:
            if self.clip_loss_version == 'clip-v0':
                loss = self.get_loss_ddp(fmri_embeddings, audio_embeddings)
            elif self.clip_loss_version == 'clip-openai':
                loss = self.get_loss_openai_ddp(fmri_embeddings, audio_embeddings)
        elif dist_config:
            loss = self.get_loss(fmri_embeddings, audio_embeddings)
            dist_config.all_reduce(loss, op=dist_config.ReduceOp.SUM)
            loss /= self.world_size
        else:
            if self.clip_loss_version == 'clip-v0':
                loss = self.get_loss(fmri_embeddings, audio_embeddings)
            elif self.clip_loss_version == 'clip-openai':
                loss = self.get_loss_openai(fmri_embeddings, audio_embeddings)
        return loss



class CLIPAudioVideo(nn.Module):

    def __init__(
        self,
        audio_model,
        video_model, 
        audio_encoder_trainable = False, 
        video_encoder_trainable = False, 
        clip_embedding_dimension = 256,
        audio_embedding_dims= 768,
        video_embedding_dims= 768,
        dropout=0.0,
        temperature = 1.0,
        device = 'cuda:0',

    ):
        super().__init__()
        
        self.device = device

        #set trainable parameters for video encoder:
        self.video_encoder = VideoEncoder(model_name=video_model, 
                                        trainable=video_encoder_trainable,
                                        device= device)

        #set trainable parameters for video encoder:
        self.audio_encoder = AudioEncoder(model_name=audio_model, 
                                        trainable=audio_encoder_trainable,
                                        device= device)
        
        #video wrapper
        self.video_wrapper = Wrapper(embedding_dim=video_embedding_dims,
                                    projection_dim=clip_embedding_dimension,
                                    dropout=dropout)
        
        self.video_wrapper.to(device)
        
        #audio wrapper
        self.audio_wrapper = Wrapper(embedding_dim=audio_embedding_dims,
                                    projection_dim=clip_embedding_dimension,
                                    dropout=dropout)
    
        self.audio_wrapper.to(device)
        
        self.temperature = temperature

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encoder_video(self,video):

        return self.video_encoder(video,self.device)

    def encoder_audio(self,audio):

        return self.audio_encoder(audio)

    def proj_video(self,video_features):

        return self.video_wrapper(video_features)
    
    def proj_audio(self,audio_features):

        return self.audio_wrapper(audio_features)

    def get_loss(self,fmri_embeddings, audio_embeddings):

        # Calculating the Loss
        logits = (audio_embeddings @ fmri_embeddings.T) / self.temperature
        fmri_similarity = fmri_embeddings @ fmri_embeddings.T
        audio_similarity = audio_embeddings @ audio_embeddings.T
        targets = F.softmax(
            (fmri_similarity + audio_similarity) / 2 * self.temperature, dim=-1
        )
        audio_loss = cross_entropy(logits, targets, reduction='none')
        fmri_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (audio_loss + fmri_loss) / 2.0 # shape: (batch_size)

        return loss.mean()

    def forward(self,audio_clips,video_clips):

        video_features = self.encoder_video(video_clips)
      
        audio_features = self.encoder_audio(audio_clips)
        
        video_embeddings = self.proj_video(video_features)
       
        audio_embeddings = self.proj_audio(audio_features)
      
        loss = self.get_loss(video_embeddings, audio_embeddings)
        return loss

def cross_entropy(preds, targets, reduction='none'):

    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
    


class VideoEncoder(nn.Module):
    def __init__(self,
                model_name, 
                trainable,
                device):
        
        super().__init__()

        if model_name == 'videomae':
            config_path = './mmaction2/configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400-custom.py'
            checkpoint_path = 'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth'

        self.model = init_recognizer(config_path, checkpoint_path, device="{}".format(device))  # device can be 'cuda:0'
        print('')
        self.custom_cfg = self.model.cfg.test_pipeline

        for p in self.model.parameters():
            p.requires_grad = trainable


    def forward(self, movie_clips,device):
                        
        self.activations = {}
        self.model.backbone.fc_norm.register_forward_hook(self.hook_fn)
        embeddings = []

        for clip in movie_clips:
            _ = inference_recognizer(self.model, clip, test_pipeline=Compose(self.custom_cfg))
            embeddings.append(self.activations['value'].squeeze().to(device))

        embeddings = torch.stack(embeddings,dim=0).to(device)

        return embeddings

    def hook_fn(self, module, input, output):
        self.activations['value'] = output



class AudioEncoder(nn.Module):
    def __init__(self,
                model_name, 
                trainable,
                device):
        
        super().__init__()

        if model_name == 'wav2vec2':
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    
        print("Sample Rate:", bundle.sample_rate)
        print("Labels:", bundle.get_labels())

        self.model = bundle.get_model().to(device)

        for p in self.model.parameters():
            p.requires_grad = trainable


    def forward(self, audio_segments, device):


        #print(audio_segments.shape)
        with torch.inference_mode():
            features, _ = self.model.extract_features(audio_segments)
        
        features = features[-1].mean(1).to(device)
        #print('out features shape: {}'.format(features))

        return features

class fMRIEncoder(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        trainable,
        use_class_token=False, 

    ):
        super().__init__()
        
        self.encoder = encoder
        self.use_class_token = use_class_token
                
        for p in self.encoder.parameters():
            p.requires_grad=trainable
        
        self.num_patches_encoder, _ = self.encoder.num_patches, self.encoder.encoding_dim
        #self.to_patch, self.patch_to_emb = self.encoder.to_patch_embedding[:2]
        #self.norm = encoder.mlp_head[0]

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder.encoding_dim)) if use_class_token else None
       
    def forward_encoder(self, img):
    
        # get patches
        patches = self.encoder.to_patch_embedding[0](img)
        B, num_patches, *_ = patches.shape

        assert (num_patches == self.num_patches_encoder)

        # patch to encoder tokens and add positions
        x = self.encoder.to_patch_embedding[1](patches)
                
        if self.encoder.no_class_token_emb:
            x = x + self.encoder.pos_embedding[:,:num_patches,:]  #can be set to fixed in the encoder 
        else:
            raise NotImplementedError
        #else:
        #    x = x + self.encoder.pos_embedding[:,self.encoder.num_prefix_tokens:(num_patches+self.encoder.num_prefix_tokens),:] #use use class toekn: 1-> n+1 else, 0->n
            
        # append NEW cls token
        if self.use_class_token:
            cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = B)
            x = torch.cat((cls_tokens, x), dim=1)    
        
        x = self.encoder.transformer(x)
                
        x = self.encoder.mlp_head[0](x)
                
        return x

    def forward(self,x):
        return self.forward_encoder(x)
    
    