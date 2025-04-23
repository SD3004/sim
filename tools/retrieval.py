import argparse
import os
import numpy as np
import nibabel as nb
import torch
import random
import time
import itertools
from einops import rearrange, reduce
import h5py
import torch.nn.functional as F
import yaml
import pandas as pd
import sys
import matplotlib.pyplot as plt
import datetime

sys.path.append('/home/sd20/workspace/fMRI_transformers/')


from utils import get_dimensions

from models.sit import SiT
from models.clip import *
from models.MLP import SimpleMLP


from moviepy.editor import VideoFileClip


fmri_session_structure = {}
fmri_session_structure['MOVIE1_CC1'] = {}
fmri_session_structure['MOVIE1_CC1']['rest_period_1'] = [0,20]
fmri_session_structure['MOVIE1_CC1']['movie_sample_1'] = [21,263]
fmri_session_structure['MOVIE1_CC1']['rest_period_2'] = [264,284]
fmri_session_structure['MOVIE1_CC1']['movie_sample_2'] = [285,504]
fmri_session_structure['MOVIE1_CC1']['rest_period_3'] = [505,525]
fmri_session_structure['MOVIE1_CC1']['movie_sample_3'] = [526,713]
fmri_session_structure['MOVIE1_CC1']['rest_period_4'] = [714,733]
fmri_session_structure['MOVIE1_CC1']['movie_sample_4'] = [734,797]
fmri_session_structure['MOVIE1_CC1']['rest_period_5'] = [798,817]
fmri_session_structure['MOVIE1_CC1']['test_retest'] = [818,900]
fmri_session_structure['MOVIE1_CC1']['rest_period_6'] = [901,920]

fmri_session_structure['MOVIE2_HO1'] = {}
fmri_session_structure['MOVIE2_HO1']['rest_period_1'] = [0,20]
fmri_session_structure['MOVIE2_HO1']['movie_sample_1'] = [21,246]
fmri_session_structure['MOVIE2_HO1']['rest_period_2'] = [247,266]
fmri_session_structure['MOVIE2_HO1']['movie_sample_2'] = [267,525]
fmri_session_structure['MOVIE2_HO1']['rest_period_3'] = [526,544]
fmri_session_structure['MOVIE2_HO1']['movie_sample_3'] = [545,794]
fmri_session_structure['MOVIE2_HO1']['rest_period_4'] = [795,814]
fmri_session_structure['MOVIE2_HO1']['test_retest'] = [815,897]
fmri_session_structure['MOVIE2_HO1']['rest_period_5'] = [898,917]

fmri_session_structure['MOVIE3_CC2'] = {}
fmri_session_structure['MOVIE3_CC2']['rest_period_1'] = [0,20]
fmri_session_structure['MOVIE3_CC2']['movie_sample_1'] = [21,199]
fmri_session_structure['MOVIE3_CC2']['rest_period_2'] = [200,220]
fmri_session_structure['MOVIE3_CC2']['movie_sample_2'] = [221,404]
fmri_session_structure['MOVIE3_CC2']['rest_period_3'] = [405,424]
fmri_session_structure['MOVIE3_CC2']['movie_sample_3'] = [425,628]
fmri_session_structure['MOVIE3_CC2']['rest_period_4'] = [629,649]
fmri_session_structure['MOVIE3_CC2']['movie_sample_4'] = [650,791]
fmri_session_structure['MOVIE3_CC2']['rest_period_5'] = [792,811]
fmri_session_structure['MOVIE3_CC2']['test_retest'] = [812,894]
fmri_session_structure['MOVIE3_CC2']['rest_period_6'] = [895,914]

fmri_session_structure['MOVIE4_HO2'] = {}
fmri_session_structure['MOVIE4_HO2']['rest_period_1'] = [0,20]
fmri_session_structure['MOVIE4_HO2']['movie_sample_1'] = [21,252]
fmri_session_structure['MOVIE4_HO2']['rest_period_2'] = [253,271]
fmri_session_structure['MOVIE4_HO2']['movie_sample_2'] = [272,501]
fmri_session_structure['MOVIE4_HO2']['rest_period_3'] = [502,521]
fmri_session_structure['MOVIE4_HO2']['movie_sample_3'] = [522,777]
fmri_session_structure['MOVIE4_HO2']['rest_period_4'] = [778,797]
fmri_session_structure['MOVIE4_HO2']['test_retest'] = [798,880]
fmri_session_structure['MOVIE4_HO2']['rest_period_5'] = [881,900]


class CLIPInference():
    
    def __init__(self, model, 
                 config,
                 device,
                 mask,
                 triangle_indices,
                 path_to_data,
                 clip_duration,
                 split_to_test,
                 split_info,
                 stimulus='movie',
                 exclude_positive_rest_from_eval=False,
                 exclude_negative_rest_from_eval=False,
                 fmri_encoder_model='sit'):
        
        self.model = model
        self.device = device
        self.triangle_indices = triangle_indices
        self.mask=mask
        self.sub_ico = config['mesh_resolution']['ico_grid']
        self.nbr_patches = config['ico_{}_grid'.format(self.sub_ico)]['num_patches']
        self.nbr_vertices = config['ico_{}_grid'.format(self.sub_ico)]['num_vertices']
        self.split_to_test = split_to_test
        self.df = split_info
        self.config = config
        self.stimulus = stimulus
        self.exclude_positive_rest_from_eval = exclude_positive_rest_from_eval
        self.exclude_negative_rest_from_eval = exclude_negative_rest_from_eval
        self.fmri_encoder_model  = fmri_encoder_model

        if stimulus == 'video':
            file_path = '{}/HCP/movies/Pre_20140821_version/video_embeddings/{}s/embeddings_movie_1.h5'.format(path_to_data,clip_duration)
            with h5py.File(file_path, 'r') as f:
                self.movie_embeddings_1 = f['final']['activations'][:].squeeze()
            file_path = '{}/HCP/movies/Pre_20140821_version/video_embeddings/{}s/embeddings_movie_2.h5'.format(path_to_data,clip_duration)
            with h5py.File(file_path, 'r') as f:
                self.movie_embeddings_2 = f['final']['activations'][:].squeeze()
            file_path = '{}/HCP/movies/Pre_20140821_version/video_embeddings/{}s/embeddings_movie_3.h5'.format(path_to_data,clip_duration)
            with h5py.File(file_path, 'r') as f:
                self.movie_embeddings_3 = f['final']['activations'][:].squeeze()
            file_path = '{}/HCP/movies/Pre_20140821_version/video_embeddings/{}s/embeddings_movie_4.h5'.format(path_to_data,clip_duration)
            with h5py.File(file_path, 'r') as f:
                self.movie_embeddings_4 = f['final']['activations'][:].squeeze()
        elif stimulus == 'audio':
            file_path = '{}/HCP/movies/Pre_20140821_version/audio_embeddings/{}s/embeddings_audio_movie_1.h5'.format(path_to_data,clip_duration)
            with h5py.File(file_path, 'r') as f:
                self.audio_embeddings_1 = f['final']['activations'][:].squeeze()
            file_path = '{}/HCP/movies/Pre_20140821_version/audio_embeddings/{}s/embeddings_audio_movie_2.h5'.format(path_to_data,clip_duration)
            with h5py.File(file_path, 'r') as f:
                self.audio_embeddings_2 = f['final']['activations'][:].squeeze()
            file_path = '{}/HCP/movies/Pre_20140821_version/audio_embeddings/{}s/embeddings_audio_movie_3.h5'.format(path_to_data,clip_duration)
            with h5py.File(file_path, 'r') as f:
                self.audio_embeddings_3 = f['final']['activations'][:].squeeze()
            file_path = '{}/HCP/movies/Pre_20140821_version/audio_embeddings/{}s/embeddings_audio_movie_4.h5'.format(path_to_data,clip_duration)
            with h5py.File(file_path, 'r') as f:
                self.audio_embeddings_4 = f['final']['activations'][:].squeeze()
        
    def select_frames(self,t0, n=3):
        return [t0+i for i in range(n)]
    
    def load_fmri_image(self,
                        t,
                        idx,
                        filedir = '/drive/data/HCP/movie_frames/',
                        ):
        path = os.path.join(filedir,
                        str(self.df['ids'].iloc[idx]),
                        'tfMRI_MOVIE{}_7T_{}'.format(self.df['movie'][idx],self.df['session'][idx]),
                        'frames_ico6_{}'.format(self.df['hemi'][idx]),
                        'frame_{}.{}.shape.gii'.format(str(t).zfill(3),self.df['hemi'][idx]))
        data =  np.array(nb.load(path).agg_data())

        if len(data.shape)==1:
            data = np.expand_dims(data,0)

        return data.squeeze()

        
    def load_fmri_clip(self,
                    t0,
                    clip_duration,
                    idx):

        frames_to_extract = self.select_frames(t0,n=clip_duration)

        data = []
        for f in frames_to_extract:
            data.append(self.load_fmri_image(f,idx))

        data = np.stack(data,axis=0)

        return data

    def normalise_frames(self,
                         data, 
                        mask):
        
        non_masked_vertices = mask>0
        data[:,non_masked_vertices] = (data[:,non_masked_vertices] - data[:,non_masked_vertices].mean(axis=1).reshape(data.shape[0],1))/data[:,non_masked_vertices].std(axis=1).reshape(data.shape[0],1)
        return data

    def get_sequence(self,
                     data,
                    n,
                    triangle_indices):
        
        
        sequence = np.zeros((n, self.nbr_patches, self.nbr_vertices))
        for j in range(self.nbr_patches):
            indices_to_extract = triangle_indices[str(j)].to_numpy()
            sequence[:,j,:] = data[:,indices_to_extract]
        return sequence


    def temporal_mixing(self,
                        sequence,
                        ):

            #print('******* temporal mixing - {} *******:'.format(self.temporal_rep))
            if self.config['fMRI']['temporal_rep'] == 'concat':

                sequence = rearrange(sequence, 't n v -> 1 (t n) v') #concat frames
            
            elif self.config['fMRI']['temporal_rep'] == 'avg':

                sequence = reduce(sequence, 't n v -> 1 n v', 'mean')

            elif self.config['fMRI']['temporal_rep'] == 'channels':

                return sequence

            else:
                raise('Not implemented yet')
            
            return sequence


    def load_fmri_sequence(self,
                           t0,
                            clip_duration,
                            idx,
                            ):

        data = self.load_fmri_clip(
                                t0,
                                clip_duration,
                                idx)

        data = self.normalise_frames(data,self.mask)

        data = self.get_sequence(data,3,self.triangle_indices)

        sequence = self.temporal_mixing(data)

        return torch.from_numpy(sequence).float()
    
    
    def get_stimulus_clip_info(self,
                            t,
                            idx,
                            clip_duration=3,
                            return_video_frames=False
                            ):

        movie = self.df.iloc[idx].movie
        nbr_frames = self.df.iloc[idx]['nbr_frames']
        video_frames = []

        if movie==1:
            movie_name= 'MOVIE1_CC1'
        elif movie ==2:
            movie_name = 'MOVIE2_HO1'
        elif movie ==3:
            movie_name = 'MOVIE3_CC2'
        else:
            movie_name = 'MOVIE4_HO2'

        t0_delayed = max(0, t)
        clip_id = str(t0_delayed).zfill(3)

        if self.stimulus == 'video':
                
            video_clip = '/home/sd20/data/HCP/movies/Pre_20140821_version/clips_7T_{}/{}s/clip_{}.mp4'.format(movie_name,clip_duration,clip_id)

            if return_video_frames:
                clip = VideoFileClip(video_clip)
                # Assuming all clips are 3 seconds long,
                # iterate through each second and save the frame
                for f in range(clip_duration):  # 0, 1, 2, 3 seconds
                    video_frames.append(clip.get_frame(f))  # Get the frame at each second
                
                clip.close()

            return video_clip, movie_name, nbr_frames, clip_id,  np.array(video_frames)

        elif self.stimulus  == 'audio':
            return movie_name, nbr_frames, clip_id
    
    def get_inter_available_ids(self,
                                ref_frame, 
                                movie_name,
                                movie_sampling_strategy,
                                ):
            
        for clip_video, ids in fmri_session_structure[str(movie_name)].items():
            if ref_frame in list(range(ids[0], ids[1] +1)):
                ref_clip_sample = clip_video

        available_ids = []
        for clip_video, ids in fmri_session_structure[str(movie_name)].items():
            if clip_video != ref_clip_sample:
                if (not self.exclude_negative_rest_from_eval) or (self.exclude_negative_rest_from_eval and ('rest' not in clip_video)):
                    if movie_sampling_strategy == 'any':
                        available_ids.append(list(range(ids[0], ids[1] +1)))
                    elif movie_sampling_strategy == 'half':
                        if 'movie' in ref_clip_sample:
                            # Determine the range for sampling in the second half
                            start, end = ids
                            half_point = (start + end) // 2
                            sample_range = (half_point + 1, end)
                            available_ids.append(list(range(sample_range[0], sample_range[1] +1)))
                        else:
                            available_ids.append(list(range(ids[0], ids[1] +1)))

                    else:
                        raise NotImplementedError
            
        available_ids = list(itertools.chain(*available_ids))
        
        return available_ids
    
    def get_intra_available_ids(self,
                                ref_frame, 
                                movie_name,
                                movie_sampling_strategy):

        
        for clip_video, ids in fmri_session_structure[str(movie_name)].items():
            if ref_frame in list(range(ids[0], ids[1] +1)):
                ref_clip_sample = clip_video
        
        ids = fmri_session_structure[str(movie_name)][ref_clip_sample]

        if movie_sampling_strategy == 'half' and 'movie' in ref_clip_sample:
            # Determine the range for sampling in the second half
            start, end = ids
            half_point = (start + end) // 2
            ids = (half_point + 1, end)    
                
        return list(range(ids[0], ids[1] +1))

    
    def get_other_video_clip_infos(self,
                                   ref_frame,
                                    movie_name,
                                    clip_duration,
                                    nbr_frames,
                                    nbr_stimuli_to_sample=3,
                                    sampling_strategy = 'intra_movie_sample', # 'intra_movie_sample' or 'inter_movie_sample' or all 
                                    sampling_type = '',
                                    movie_sampling_strategy = 'any',
                                    return_video_frames=False,
                                    ): # 'random' or 'uniform'

        video_clips = []
        clip_ids = []
        video_frames_clips = []

        if sampling_strategy == 'inter_movie_sample':

            available_ids = self.get_inter_available_ids(ref_frame, movie_name, movie_sampling_strategy)
            if ref_frame in available_ids:
                available_ids.remove(ref_frame)
            if ref_frame-1 in available_ids:
                available_ids.remove(ref_frame-1)
            if ref_frame-2 in available_ids:
                available_ids.remove(ref_frame-2)
            if ref_frame-3 in available_ids:
                available_ids.remove(ref_frame-3)
            if ref_frame+1 in available_ids:
                available_ids.remove(ref_frame+1)
            if ref_frame+2 in available_ids:
                available_ids.remove(ref_frame+2)
            if ref_frame+3 in available_ids:
                available_ids.remove(ref_frame+3)
        
        elif sampling_strategy == 'intra_movie_sample':

            available_ids = self.get_intra_available_ids(ref_frame, movie_name, movie_sampling_strategy)
            if ref_frame in available_ids:
                available_ids.remove(ref_frame)
            if ref_frame-1 in available_ids:
                available_ids.remove(ref_frame-1)
            if ref_frame-2 in available_ids:
                available_ids.remove(ref_frame-2)
            if ref_frame-3 in available_ids:
                available_ids.remove(ref_frame-3)
            if ref_frame+1 in available_ids:
                available_ids.remove(ref_frame+1)
            if ref_frame+2 in available_ids:
                available_ids.remove(ref_frame+2)
            if ref_frame+3 in available_ids:
                available_ids.remove(ref_frame+3)

        elif sampling_strategy == 'all':

            available_ids = range(0, nbr_frames-clip_duration-1)
        
        if sampling_type == 'random':
            # sample randomly from all other movie clips   
            sampled_ids = random.sample(available_ids, nbr_stimuli_to_sample)

            for clip_id in sampled_ids:
                clip_id = min(clip_id, nbr_frames-clip_duration-1)
                clip_ids.append(clip_id)
                video_clip = '/home/sd20/data/HCP/movies/Pre_20140821_version/clips_7T_{}/{}s/clip_{}.mp4'.format(movie_name,
                                                                                                            clip_duration,
                                                                                                            str(clip_id).zfill(3))
                video_clips.append(video_clip)

                if return_video_frames:

                    video_frames = []
                    clip = VideoFileClip(video_clip)

                    for f in range(clip_duration):  # 0, 1, 2, 3 seconds
                        video_frames.append(clip.get_frame(f))  # Get the frame at each second
                    
                    clip.close()
                    
                    video_frames_clips.append(np.array(video_frames))
        
        elif sampling_type == 'uniform':
            # sample uniformly from all other movie clips   
            sampled_ids = self.uniform_sampling_indices(available_ids, nbr_stimuli_to_sample)

            for clip_id in sampled_ids:
                clip_id = min(clip_id, nbr_frames-clip_duration-1)
                clip_ids.append(clip_id)
                video_clip = '/home/sd20/data/HCP/movies/Pre_20140821_version/clips_7T_{}/{}s/clip_{}.mp4'.format(movie_name,
                                                                                                            clip_duration,
                                                                                                            str(clip_id).zfill(3))
                video_clips.append(video_clip)

                if return_video_frames:

                    video_frames = []
                    clip = VideoFileClip(video_clip)

                    for f in range(clip_duration):  # 0, 1, 2, 3 seconds
                        video_frames.append(clip.get_frame(f))  # Get the frame at each second
                    
                    clip.close()
                    
                    video_frames_clips.append(np.array(video_frames))

        return video_clips, clip_ids, video_frames_clips
    
    def get_other_audio_clip_infos(self,
                                   ref_frame,
                                    movie_name,
                                    clip_duration,
                                    nbr_frames,
                                    nbr_stimuli_to_sample=3,
                                    sampling_strategy = 'intra_movie_sample', # 'intra_movie_sample' or 'inter_movie_sample' or all 
                                    sampling_type = '',
                                    movie_sampling_strategy = 'any',
                                    ): # 'random' or 'uniform'

        audio_clips = []
        clip_ids = []

        if sampling_strategy == 'inter_movie_sample':

            available_ids = self.get_inter_available_ids(ref_frame, movie_name, movie_sampling_strategy)
            if ref_frame in available_ids:
                available_ids.remove(ref_frame)
        
        elif sampling_strategy == 'intra_movie_sample':

            available_ids = self.get_intra_available_ids(ref_frame, movie_name, movie_sampling_strategy)
            if ref_frame in available_ids:
                available_ids.remove(ref_frame)

        elif sampling_strategy == 'all':

            available_ids = range(0, nbr_frames-clip_duration-1)
        
        if sampling_type == 'random':
            # sample randomly from all other movie clips   
            sampled_ids = random.sample(available_ids, nbr_stimuli_to_sample)

            for clip_id in sampled_ids:
                clip_id = min(clip_id, nbr_frames-clip_duration-1)
                clip_ids.append(clip_id)
                audio_clip = '/home/sd20/data/HCP/movies/Pre_20140821_version/audio_{}/{}s/audio_clip_{}.wav'.format(movie_name,
                                                                                                            clip_duration,
                                                                                                            str(clip_id).zfill(3))
                audio_clips.append(audio_clip)
        
        elif sampling_type == 'uniform':
            # sample uniformly from all other movie clips   
            sampled_ids = self.uniform_sampling_indices(available_ids, nbr_stimuli_to_sample)

            for clip_id in sampled_ids:
                clip_id = min(clip_id, nbr_frames-clip_duration-1)
                clip_ids.append(clip_id)
                audio_clip = '/home/sd20/data/HCP/movies/Pre_20140821_version/audio_{}/{}s/audio_clip_{}.wav'.format(movie_name,
                                                                                                            clip_duration,
                                                                                                            str(clip_id).zfill(3))
                audio_clips.append(audio_clip)

        return audio_clips, clip_ids

    def load_fmri_and_movie_clips(self, 
                                  ref_video_frame,
                              temporal_lag,
                              subject_idx,
                              clip_duration=3,
                              nbr_stimuli_to_sample=3,
                              return_video_frames = False,
                              sampling_strategy = 'inter_movie_sample', # 'intra_movie_sample' or 'inter_movie_sample' or all 
                              sampling_type = 'uniform', # 'random' or 'uniform'
                              movie_sampling_strategy = 'any',
                              ):
        

        ### select a time-point in fMRI 
        ref_fmri_data = self.load_fmri_sequence(ref_video_frame + temporal_lag, # delay the fMRI signal to match the video
                                                clip_duration,
                                                subject_idx,
                                                )
            
        ref_video_clip, movie_name, nbr_frames_movie, ref_clip_id, ref_video_frames = self.get_stimulus_clip_info(ref_video_frame,
                                                                                                                subject_idx,
                                                                                                                clip_duration=clip_duration,
                                                                                                                return_video_frames=return_video_frames)  
        
        other_video_clips, other_clip_ids, other_clips_video_frames = self.get_other_video_clip_infos(ref_video_frame,
                                                                        movie_name,
                                                                        clip_duration,
                                                                        nbr_frames_movie,
                                                                        nbr_stimuli_to_sample=nbr_stimuli_to_sample,
                                                                        sampling_strategy = sampling_strategy, 
                                                                        sampling_type = sampling_type,
                                                                        movie_sampling_strategy = movie_sampling_strategy,
                                                                        return_video_frames=return_video_frames,
                                                                        )    
        

        return ref_fmri_data, ref_clip_id, ref_video_frames, other_clip_ids, other_clips_video_frames, movie_name
    
    def load_fmri_and_audio_clips(self, 
                                  ref_video_frame,
                                    temporal_lag,
                                    subject_idx,
                                    clip_duration=3,
                                    nbr_stimuli_to_sample=3,
                                    sampling_strategy = 'inter_movie_sample', # 'intra_movie_sample' or 'inter_movie_sample' or all 
                                    sampling_type = 'uniform', # 'random' or 'uniform'
                                    movie_sampling_strategy = 'any'
                                    ):
            

        ### select a time-point in fMRI 
        ref_fmri_data = self.load_fmri_sequence(ref_video_frame + temporal_lag, # delay the fMRI signal to match the video
                                                clip_duration,
                                                subject_idx,
                                                )
            
        movie_name, nbr_frames_movie, ref_clip_id = self.get_stimulus_clip_info(ref_video_frame,
                                                                                subject_idx,
                                                                                clip_duration=clip_duration,
                                                                                )  

        other_audio_clips, other_clip_ids = self.get_other_audio_clip_infos(ref_video_frame,
                                                                        movie_name,
                                                                        clip_duration,
                                                                        nbr_frames_movie,
                                                                        nbr_stimuli_to_sample=nbr_stimuli_to_sample,
                                                                        sampling_strategy = sampling_strategy, 
                                                                        sampling_type = sampling_type,
                                                                        movie_sampling_strategy = movie_sampling_strategy,
                                                                        )    
        

        return ref_fmri_data, ref_clip_id, other_clip_ids, other_audio_clips, movie_name
    
    def get_stimulus_embeddings(self,
                            movie_name, 
                            ref_stimulus_clip_id,
                            other_stimulus_clips_id):
        
        if self.stimulus == 'video':
        
            if movie_name == 'MOVIE1_CC1':
                ref_stimulus_features = torch.from_numpy(self.movie_embeddings_1[int(ref_stimulus_clip_id),:]).float().view(1,-1)
                other_stimulus_features = torch.from_numpy(self.movie_embeddings_1[other_stimulus_clips_id,:]).float()
            elif movie_name == 'MOVIE2_HO1':
                ref_stimulus_features = torch.from_numpy(self.movie_embeddings_2[int(ref_stimulus_clip_id),:]).float().view(1,-1)
                other_stimulus_features = torch.from_numpy(self.movie_embeddings_2[other_stimulus_clips_id,:]).float()
            elif movie_name == 'MOVIE3_CC2':
                ref_stimulus_features = torch.from_numpy(self.movie_embeddings_3[int(ref_stimulus_clip_id),:]).float().view(1,-1)
                other_stimulus_features = torch.from_numpy(self.movie_embeddings_3[other_stimulus_clips_id,:]).float()
            elif movie_name == 'MOVIE4_HO2':
                ref_stimulus_features = torch.from_numpy(self.movie_embeddings_4[int(ref_stimulus_clip_id),:]).float().view(1,-1)
                other_stimulus_features = torch.from_numpy(self.movie_embeddings_4[other_stimulus_clips_id,:]).float()

            ref_stimulus_emb = self.model.proj_video(ref_stimulus_features.to(self.model.device))
            other_stimulus_emb = self.model.proj_video(other_stimulus_features.to(self.model.device))  
        
        elif self.stimulus == 'audio':
        
            if movie_name == 'MOVIE1_CC1':
                ref_stimulus_features = torch.from_numpy(self.audio_embeddings_1[int(ref_stimulus_clip_id),:]).float().view(1,-1)
                other_stimulus_features = torch.from_numpy(self.audio_embeddings_1[other_stimulus_clips_id,:]).float()
            elif movie_name == 'MOVIE2_HO1':
                ref_stimulus_features = torch.from_numpy(self.audio_embeddings_2[int(ref_stimulus_clip_id),:]).float().view(1,-1)
                other_stimulus_features = torch.from_numpy(self.audio_embeddings_2[other_stimulus_clips_id,:]).float()
            elif movie_name == 'MOVIE3_CC2':
                ref_stimulus_features = torch.from_numpy(self.audio_embeddings_3[int(ref_stimulus_clip_id),:]).float().view(1,-1)
                other_stimulus_features = torch.from_numpy(self.audio_embeddings_3[other_stimulus_clips_id,:]).float()
            elif movie_name == 'MOVIE4_HO2':
                ref_stimulus_features = torch.from_numpy(self.audio_embeddings_4[int(ref_stimulus_clip_id),:]).float().view(1,-1)
                other_stimulus_features = torch.from_numpy(self.audio_embeddings_4[other_stimulus_clips_id,:]).float()

            ref_stimulus_emb = self.model.proj_audio(ref_stimulus_features.to(self.model.device))
            other_stimulus_emb = self.model.proj_audio(other_stimulus_features.to(self.model.device))  

        return ref_stimulus_emb, other_stimulus_emb
    
    def find_matches(self,
                     ref_fmri_data,
                     ref_video_emb,
                     other_video_emb,
                     topk=3):
                

        if self.fmri_encoder_model == 'sit':
            ref_fmri_features = self.model.encoder_fmri(ref_fmri_data.unsqueeze(0))
            if self.model.use_class_token_fmri_encoder:
                ref_fmri_features = ref_fmri_features[:,1:,:]
                
            if self.model.avg == 'prior_proj':
                fmri_embeddings = ref_fmri_features.mean(dim=1)
                fmri_embeddings = self.model.proj_fmri(fmri_embeddings)
            elif self.model.avg == 'post_proj':
                fmri_embeddings = self.model.proj_fmri(ref_fmri_features)
                fmri_embeddings = fmri_embeddings.mean(dim=1)
        elif self.fmri_encoder_model == 'mlp':
            #import pdb;pdb.set_trace()
            fmri_embeddings = self.model.encoder_fmri(rearrange(ref_fmri_data.unsqueeze(0), 'b n v t -> b (n v t)'))

        video_embeddings = torch.concat([ref_video_emb.to(self.model.device),other_video_emb.to(self.model.device)])
        video_embeddings_n = F.normalize(video_embeddings, p=2, dim=-1)
        fmri_embeddings_n = F.normalize(fmri_embeddings, p=2, dim=-1)
        dot_similarity = fmri_embeddings_n @ video_embeddings_n.T

        values_top, indices_top = torch.topk(dot_similarity.squeeze(0),topk)

        values_small, indices_small = torch.topk(dot_similarity.squeeze(0),topk, largest=False)

        return list(indices_top.cpu().numpy()), list(indices_small.cpu().numpy())

    def display_top_frames(self, nbr_frames, reference_video, comparison_videos, indices_top, indices_small, t, id, output_dir, top_to_display=3):
        # Concatenate the reference video with the comparison videos for easier indexing
        all_videos = [reference_video] + comparison_videos

        # Calculate the number of columns based on the top_to_display value
        num_columns = 1 + top_to_display * 2

        # Adjust figure size dynamically based on the number of columns
        fig, axs = plt.subplots(3, num_columns, figsize=(5 * num_columns, 15))  # 3 rows and num_columns columns

        # Displaying reference video frames
        for i in range(min(nbr_frames, 3)):  # Ensure we only loop through the available frames
            axs[i, 0].imshow(all_videos[0][i])  # Reference video is the first in all_videos
            axs[i, 0].axis('off')
        axs[0, 0].set_title("Reference Video", fontsize=14)

        # Display top N videos
        start_index = 1
        for j in range(top_to_display):
            video_idx = indices_top[j]
            top_video = all_videos[video_idx]
            for k in range(min(nbr_frames, 3)):  # Ensure we only loop through the available frames
                axs[k, start_index + j].imshow(top_video[k])
                axs[k, start_index + j].axis('off')
                if k == 0:
                    axs[k, start_index + j].set_title(f"Top {j + 1}", fontsize=14)

        # Display last N videos
        end_index = 1 + top_to_display
        for j in range(top_to_display):
            video_idx = indices_small[-(j + 1)]  # Get last top_to_display in reversed order
            bottom_video = all_videos[video_idx]
            for k in range(min(nbr_frames, 3)):  # Ensure we only loop through the available frames
                axs[k, end_index + j].imshow(bottom_video[k])
                axs[k, end_index + j].axis('off')
                if k == 0:
                    axs[k, end_index + j].set_title(f"Bottom {top_to_display - j}", fontsize=14)

        plt.tight_layout()

        # Save the plot to a file
        output_dir = os.path.join(output_dir, 'top_frames')
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        filename = f"{output_dir}/results_{id}_{t}.png"
        plt.savefig(filename)
        plt.close(fig)  # Close the plot to free up memory

        print(f"Plot saved to {filename}")

    
    def compute_topk_accuracy(self,
                            ids_to_test,
                            temporal_lag,
                            nbr_clip_to_test=10,
                            clip_duration=3,
                            nbr_stimuli_to_sample_per_test=6,
                            sampling_strategy = 'inter_movie_sample', # 'intra_movie_sample' or 'inter_movie_sample' or all 
                            sampling_type = 'uniform', # 'random' or 'uniform',
                            movie_sampling_strategy = 'any',
                            display=False,
                            single_subject=False,
                            topk=10,
                            topk_display=3,
                            output_dir=None,):


        correct_top_1 = 0
        correct_top_5 = 0
        correct_top_10 = 0
                
        for t in range(nbr_clip_to_test):

            movie = self.df.iloc[ids_to_test].movie
            if movie==1:
                movie_name= 'MOVIE1_CC1'
            elif movie ==2:
                movie_name = 'MOVIE2_HO1'
            elif movie ==3:
                movie_name = 'MOVIE3_CC2'
            else:
                movie_name = 'MOVIE4_HO2'   

            if movie_sampling_strategy=='any':

                if sampling_strategy == 'intra_movie_sample':
                    session_info = fmri_session_structure[str(movie_name)]
                    movie_samples = [session_info[key] for key in session_info if 'movie_sample' in key]
                    chosen_segment = random.choice(movie_samples)
                    reference_frame_id = random.randint(chosen_segment[0], chosen_segment[1])
                else:
                    if self.exclude_positive_rest_from_eval: 
                        session_info = fmri_session_structure[str(movie_name)]
                        movie_samples = [session_info[key] for key in session_info if 'movie_sample' in key]
                        chosen_segment = random.choice(movie_samples)
                        reference_frame_id = random.randint(chosen_segment[0], chosen_segment[1])
                    else:
                        reference_frame_id  = random.randint(0, self.df.iloc[ids_to_test]['nbr_frames']- 5*clip_duration)

            elif movie_sampling_strategy == 'half': 
                   
                if self.split_to_test == 'train':
                    session_info = fmri_session_structure[str(movie_name)]
                    movie_samples = [session_info[key] for key in session_info if 'movie_sample' in key]
                    chosen_segment = random.choice(movie_samples)
                    
                    start, end = chosen_segment
                    half_point = (start + end) // 2
                    sample_range = (start, half_point)
                    reference_frame_id = random.randint(sample_range[0], sample_range[1])
                
                elif self.split_to_test == 'val':
                    session_info = fmri_session_structure[str(movie_name)]
                    movie_samples = [session_info[key] for key in session_info if 'movie_sample' in key]
                    chosen_segment = random.choice(movie_samples)
                    
                    start, end = chosen_segment
                    half_point = (start + end) // 2
                    sample_range = (half_point + 1, end)
                    reference_frame_id = random.randint(sample_range[0], sample_range[1])
                
                elif self.split_to_test == 'train-for-val':
                    session_info = fmri_session_structure[str(movie_name)]
                    movie_samples = [session_info[key] for key in session_info if 'movie_sample' in key]
                    chosen_segment = random.choice(movie_samples)
                    
                    start, end = chosen_segment
                    half_point = (start + end) // 2
                    sample_range = (half_point + 1, end)
                    reference_frame_id = random.randint(sample_range[0], sample_range[1])

            
            if self.stimulus == 'video':
                
                ref_fmri_data, ref_clip_id, ref_video_frames, other_clip_ids, other_clips_video_frames, movie_name = self.load_fmri_and_movie_clips(reference_frame_id,
                                                                                                                                                    temporal_lag,
                                                                                                                                                    ids_to_test,
                                                                                                                                                    clip_duration=clip_duration,
                                                                                                                                                    nbr_stimuli_to_sample=nbr_stimuli_to_sample_per_test,
                                                                                                                                                    return_video_frames = display,
                                                                                                                                                    sampling_strategy = sampling_strategy,
                                                                                                                                                    sampling_type = sampling_type,
                                                                                                                                                    movie_sampling_strategy = movie_sampling_strategy
                                                                                                                                                    )

                

                
                
            elif self.stimulus == 'audio':


                ref_fmri_data, ref_clip_id, other_clip_ids, other_audio_clips, movie_name = self.load_fmri_and_audio_clips(reference_frame_id,
                                                                                                                            temporal_lag,
                                                                                                                            ids_to_test,
                                                                                                                            clip_duration=clip_duration,
                                                                                                                            nbr_stimuli_to_sample=nbr_stimuli_to_sample_per_test,
                                                                                                                            sampling_strategy = sampling_strategy,
                                                                                                                            sampling_type = sampling_type,
                                                                                                                            movie_sampling_strategy = movie_sampling_strategy
                                                                                                                            )
                

            ref_stimulus_emb, other_stimulus_emb = self.get_stimulus_embeddings(
                                                                    movie_name,
                                                                    ref_clip_id,
                                                                    other_clip_ids)
                
            #top 10
            indices_top, indices_small = self.find_matches( ref_fmri_data.to(self.model.device),
                                                            ref_stimulus_emb,
                                                            other_stimulus_emb,
                                                            topk=topk)
            #all_random = np.random.permutation(64).tolist()
            #import pdb;pdb.set_trace()
            #indices_top_= all_random[:topk]
            #indices_small_ = all_random[-topk:]
            #print(ref_clip_id, sorted(other_clip_ids))
            #import pdb;pdb.set_trace()
            
            if self.stimulus=='video' and display and single_subject:

                self.display_top_frames(clip_duration, 
                                        ref_video_frames,
                                          other_clips_video_frames,
                                          indices_top,
                                          indices_small,
                                          t,
                                          ids_to_test,
                                          output_dir,
                                          topk_display)


            if 0 in indices_top:
                correct_top_10+=1
            if 0 in indices_top[:5]:
                correct_top_5+=1
            if 0 == indices_top[0]:
                correct_top_1+=1

        return 100*correct_top_1/nbr_clip_to_test, 100*correct_top_5/nbr_clip_to_test, 100*correct_top_10/nbr_clip_to_test


def read_config(config_filter):
            
    # Parse the filter criteria
    filters = config_filter.split(',')
    filter_dict = {}
    for filter_item in filters:
        print(filter_item)
        column, value = filter_item.split(':')
        if column == 'sex' or column == 'movie':
            filter_dict[column] = int(value)
        else:
            filter_dict[column] = str(value)
    
    return filter_dict

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

    ###train, val data
    if config['split_to_test']=='val':
        split_info = pd.read_csv('/home/sd20/workspace/fMRI_transformers/labels/HCP/tfMRI/{}/val.csv'.format(config['train_val_split']))
        print(split_info)
    elif config['split_to_test']=='train':
        split_info = pd.read_csv('/home/sd20/workspace/fMRI_transformers/labels/HCP/tfMRI/{}/train.csv'.format(config['train_val_split']))
    elif config['split_to_test']=='train-for-val':
        split_info = pd.read_csv('/home/sd20/workspace/fMRI_transformers/labels/HCP/tfMRI/{}/train.csv'.format(config['train_val_split']))
    else:
        raise NotImplementedError
    
    if config['filter']:
        print('Filtering data: {}]'.format(config['filter']))
        filter_dict = read_config(config['filter'])
        split_info = split_info.loc[(split_info[list(filter_dict)] == pd.Series(filter_dict)).all(axis=1)]
        split_info.reset_index(inplace=True)
        print(split_info)
    
    mask = np.array(nb.load('/home/sd20/data/template_spheres/L.atlasroi.40k_fs_LR.shape.gii').agg_data())
    path_to_data = '/home/sd20/data'

    path_to_ckpt =  config['path_to_ckpt']
    device = torch.device("cuda:{}".format(config['gpu']) if torch.cuda.is_available() else "cpu")

    with open(os.path.join(path_to_ckpt,'hparams.yml')) as f:
        config_ckpt = yaml.safe_load(f)

    clip_duration = config_ckpt['clip']['clip_duration']
    ico = config_ckpt['mesh_resolution']['ico_mesh']
    sampling = config_ckpt['mesh_resolution']['sampling']
    sub_ico = config_ckpt['mesh_resolution']['ico_grid']
    triangle_indices = pd.read_csv('/home/sd20/workspace/fMRI_transformers/patch_extraction/{}/triangle_indices_ico_{}_sub_ico_{}.csv'.format(sampling,ico,sub_ico))   

    T, N, V, use_bottleneck, bottleneck_dropout = get_dimensions(config_ckpt)

    if config['fmri_encoder'] == 'sit':   

        fmri_model = SiT(dim=config_ckpt['transformer']['dim'],
                                depth=config_ckpt['transformer']['depth'],
                                heads=config_ckpt['transformer']['heads'],
                                pool=config_ckpt['transformer']['pool'], 
                                num_patches=N,
                                num_classes=config_ckpt['transformer']['num_classes'],
                                num_channels=T,
                                num_vertices=V,
                                dim_head=config_ckpt['transformer']['dim_head'],
                                dropout=config_ckpt['transformer']['dropout'],
                                emb_dropout=config_ckpt['transformer']['emb_dropout'],
                                use_pe=config_ckpt['transformer']['use_pos_embedding'],
                                bottleneck_dropout=bottleneck_dropout,
                                use_bottleneck=use_bottleneck,
                                use_confounds=False,
                                weights_layers_init=config_ckpt['transformer']['weights_layers_init'],
                                use_class_token=config_ckpt['transformer']['use_class_token'],
                                trainable_pos_emb=config_ckpt['transformer']['trainable_pos_emb'],
                                no_class_token_emb = config_ckpt['transformer']['no_class_token_emb'],)
    elif config['fmri_encoder'] == 'mlp':
        print('Using MLP')
        fmri_model = SimpleMLP(input_size=T*N*V, 
                               output_size=config_ckpt['clip']['clip_embedding_dimension'])
        
    
    if config['model'] == 'CLIP-video':
        
        model = CLIPfMRIVideo(video_model=config_ckpt['video_model']['video_name'],
                                    fmri_encoder=fmri_model,
                                    fmri_encoder_model=config_ckpt['clip']['fmri_encoder'],
                                    video_encoder_trainable=config_ckpt['clip']['video_encoder_trainable'],
                                    fmri_encoder_trainable=config_ckpt['clip']['fmri_encoder_trainable'],
                                    use_class_token_fmri_encoder=config_ckpt['clip']['use_class_token_fmri_encoder'],
                                    clip_embedding_dimension=config_ckpt['clip']['clip_embedding_dimension'],
                                    fmri_embedding_dims=config_ckpt['clip']['fmri_embedding_dims'],
                                    video_embedding_dims=config_ckpt['clip']['video_embedding_dims'],
                                    dropout=config_ckpt['clip']['dropout'],
                                    temperature=config_ckpt['clip']['temperature'],
                                    video_loader = config_ckpt['video_model']['video_loader'],
                                    avg=config_ckpt['clip']['avg_fmri'],
                                    device= device
                                    )


        
        video_wrapper_weights = torch.load(os.path.join(path_to_ckpt,'video-wrapper-best.pt'),map_location=device)
        model.video_wrapper.load_state_dict(video_wrapper_weights['model_state_dict'],strict=True)

    elif config['model'] == 'CLIP-audio':

        model = CLIPfMRIAudio(audio_model=config_ckpt['audio_model']['audio_name'],
                                fmri_encoder=fmri_model,
                                fmri_encoder_model=config_ckpt['clip']['fmri_encoder'],
                                audio_encoder_trainable=config_ckpt['clip']['audio_encoder_trainable'],
                                fmri_encoder_trainable=config_ckpt['clip']['fmri_encoder_trainable'],
                                use_class_token_fmri_encoder=config_ckpt['clip']['use_class_token_fmri_encoder'],
                                clip_embedding_dimension=config_ckpt['clip']['clip_embedding_dimension'],
                                fmri_embedding_dims=config_ckpt['clip']['fmri_embedding_dims'],
                                audio_embedding_dims=config_ckpt['clip']['audio_embedding_dims'],
                                dropout=config_ckpt['clip']['dropout'],
                                temperature=config_ckpt['clip']['temperature'],
                                audio_loader = config_ckpt['audio_model']['audio_loader'],
                                avg=config_ckpt['clip']['avg_fmri'],
                                device= device
                                )
        
        audio_wrapper_weights = torch.load(os.path.join(path_to_ckpt,'audio-wrapper-best.pt'),map_location=device)
        model.audio_wrapper.load_state_dict(audio_wrapper_weights['model_state_dict'],strict=True)

    
    elif config['model'] == 'CLIP-fmri-video-audio':

        model = CLIPfMRIVideoAudio(video_model=config_ckpt['video_model']['video_name'],
                                   audio_model=config_ckpt['audio_model']['audio_name'],
                                    fmri_encoder=fmri_model,
                                    video_encoder_trainable=config_ckpt['clip']['video_encoder_trainable'],
                                    audio_encoder_trainable=config_ckpt['clip']['audio_encoder_trainable'],
                                    fmri_encoder_trainable=config_ckpt['clip']['fmri_encoder_trainable'],
                                    use_class_token_fmri_encoder=config_ckpt['clip']['use_class_token_fmri_encoder'],
                                    clip_embedding_dimension=config_ckpt['clip']['clip_embedding_dimension'],
                                    fmri_embedding_dims=config_ckpt['clip']['fmri_embedding_dims'],
                                    video_embedding_dims=config_ckpt['clip']['video_embedding_dims'],
                                    audio_embedding_dims=config_ckpt['clip']['audio_embedding_dims'],
                                    dropout=config_ckpt['clip']['dropout'],
                                    temperature_video=config_ckpt['clip']['temperature_video'],
                                    temperature_audio=config_ckpt['clip']['temperature_audio'],
                                    video_loader = config_ckpt['video_model']['video_loader'],
                                    audio_loader = config_ckpt['audio_model']['audio_loader'],
                                    avg=config_ckpt['clip']['avg_fmri'],
                                    device= device
                                    )
        
        video_wrapper_weights = torch.load(os.path.join(path_to_ckpt,'video-wrapper-best.pt'),map_location=device)
        model.video_wrapper.load_state_dict(video_wrapper_weights['model_state_dict'],strict=True)
        audio_wrapper_weights = torch.load(os.path.join(path_to_ckpt,'audio-wrapper-best.pt'),map_location=device)
        model.audio_wrapper.load_state_dict(audio_wrapper_weights['model_state_dict'],strict=True)
        
    fmri_encoder_weights = torch.load(os.path.join(path_to_ckpt,'fmri-encoder-best.pt'),map_location=device)
    fmri_wrapper_weights = torch.load(os.path.join(path_to_ckpt,'fmri-wrapper-best.pt'),map_location=device)

    model.fmri_encoder.load_state_dict(fmri_encoder_weights['model_state_dict'],strict=True)
    model.fmri_wrapper.load_state_dict(fmri_wrapper_weights['model_state_dict'],strict=True)
    #import pdb;pdb.set_trace()
            

    model.eval()
    model.to(device)

    inference_model = CLIPInference(model, 
                                    config_ckpt,
                                    device,
                                    mask,
                                    triangle_indices,
                                    path_to_data,
                                    clip_duration,
                                    config['split_to_test'],
                                    split_info,
                                    config['stimulus'],
                                    config['exclude_positive_rest_from_eval'],
                                    config['exclude_negative_rest_from_eval'],
                                    config_ckpt['clip']['fmri_encoder']
                                    )
    
    ############## LOG ################
    output_dir = './outputs/clip_evaluation'
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(output_dir, config['model'],path_to_ckpt.split('/')[-1],config['stimulus']+'-'+config['sampling_strategy']+'-'+config['train_val_split']+'-'+config['split_to_test']+'-'+config['sampling_type']+'-exclude-rest-'+str(config['exclude_positive_rest_from_eval']), current_time)
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    print(output_dir)
    #log_file = open(os.path.join(output_dir,'log.txt'), "w")
    # Redirect standard output and standard error to the log file
    #sys.stdout = log_file
    #sys.stderr = log_file

    
    with open(os.path.join(output_dir,'hparams_ckpt.yml'), 'w') as yaml_file:
        yaml.dump(config_ckpt, yaml_file)

    
    if config['single_subject']:    

        print('*'*60)
        print('CLIP evaluation inference for subject: {}'.format(config['subject_id']))
        print('nbr_clip_to_test: {}'.format(config['nbr_clip_to_test']))
        print('nbr_stimuli_to_sample_per_test: {}'.format(config['nbr_stimuli_to_sample_per_test']))
        print('sampling_strategy: {}'.format(config['sampling_strategy']))
        print('sampling_type: {}'.format(config['sampling_type']))
        print(path_to_ckpt)
        print('stimulus to test: {}'.format(config['stimulus']))
        print('*'*60)
        
        top_1, top_5, top_10 = inference_model.compute_topk_accuracy(temporal_lag=config_ckpt['fMRI']['temporal_lag'],
                                                ids_to_test=config['subject_id'],
                                                nbr_clip_to_test=config['nbr_clip_to_test'],
                                                clip_duration=clip_duration,
                                                nbr_stimuli_to_sample_per_test=config['nbr_stimuli_to_sample_per_test'],
                                                sampling_strategy = config['sampling_strategy'], # 'intra_movie_sample' or 'inter_movie_sample' or all 
                                                sampling_type = config['sampling_type'], # 'random' or 'uniform',
                                                movie_sampling_strategy = config['movie_sampling_strategy'],
                                                display=config['display'],
                                                single_subject=True,
                                                topk=config['topk'],
                                                topk_display=config['topkdisplay'],
                                                output_dir=output_dir,)
        
        print('TOP 1 accuracy - {}%'.format(np.mean(top_1)))
        print('TOP 5 accuracy - {}%'.format(np.mean(top_5)))
        print('TOP 10 accuracy - {}%'.format(np.mean(top_10)))

        with open(os.path.join(output_dir,'hparams_eval.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)
                
    else:
        print('CLIP evaluation inference for all subjects')
        print('*'*60)
        print('nbr_clip_to_test: {}'.format(config['nbr_clip_to_test']))
        print('nbr_stimuli_to_sample_per_test: {}'.format(config['nbr_stimuli_to_sample_per_test']))
        print('sampling_strategy: {}'.format(config['sampling_strategy']))
        print('sampling_type: {}'.format(config['sampling_type']))
        print(path_to_ckpt)
        print('stimulus to test: {}'.format(config['stimulus']))
        print('*'*60)

        top_1_val = []
        top_5_val = []
        top_10_val = []
        ids = []
        movie = []
        hemi = []
        

        for i, row in split_info.iterrows():

            t0 = time.time()
            
            top_1, top_5, top_10 = inference_model.compute_topk_accuracy(temporal_lag=config_ckpt['fMRI']['temporal_lag'],
                                                    ids_to_test=i,
                                                    nbr_clip_to_test=config['nbr_clip_to_test'],
                                                    clip_duration=clip_duration,
                                                    nbr_stimuli_to_sample_per_test=config['nbr_stimuli_to_sample_per_test'],
                                                    sampling_strategy = config['sampling_strategy'], # 'intra_movie_sample' or 'inter_movie_sample' or all 
                                                    sampling_type = config['sampling_type'], # 'random' or 'uniform',
                                                    movie_sampling_strategy = config['movie_sampling_strategy'],
                                                    display=config['display'],
                                                    single_subject=False,
                                                    topk=config['topk'],
                                                    output_dir=output_dir)
            
            print('Id: {}'.format(i), 'subject {}'.format(row['ids']), 'movie {}'.format(row['movie']), 'hemi {}'.format(row['hemi']),'top 1 {}'.format(top_1), 'top 5 {}'.format(top_5), 'top 10 {}'.format(top_10), 'time {}'.format(time.time()-t0))
                                
            top_1_val.append(top_1)
            top_5_val.append(top_5)
            top_10_val.append(top_10)
            ids.append(row['ids'])
            hemi.append(row['hemi'])
            movie.append(row['movie'])
        
        print('TOP 1 accuracy - {}%'.format(np.mean(top_1_val)))
        print('TOP 5 accuracy - {}%'.format(np.mean(top_5_val)))
        print('TOP 10 accuracy - {}%'.format(np.mean(top_10_val)))
        
        df = pd.DataFrame({
                            'ids':ids,
                            'hemi':hemi,
                            'movie':movie,
                            'top_1':top_1_val,
                            'top_5':top_5_val,
                            'top_10':top_10_val,})
        df.to_csv(os.path.join(output_dir,'results.csv'))  
        
        config['top_1_mean'] = str(np.mean(top_1_val))
        config['top_1_std'] = str(np.std(top_1_val))
        config['top_5_mean'] = str(np.mean(top_5_val))
        config['top_5_std'] = str(np.std(top_5_val))
        config['top_10_mean'] = str(np.mean(top_10_val))
        config['top_10_std'] = str(np.std(top_10_val))
        
        with open(os.path.join(output_dir,'hparams_eval.yml'), 'w') as yaml_file:
            yaml.dump(config, yaml_file)

        #log_file.close()
                
                


