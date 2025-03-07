from logging import raiseExceptions
import os
import sys 
sys.path.append('./')
sys.path.append('../')

import numpy as np
import nibabel as nb
import pandas as pd
import random 
import h5py

import torch
from torch.utils.data import Dataset

from scipy.spatial.transform import Rotation as R

from einops import rearrange, reduce 

import torchaudio

# Filter out the specific warning
#warnings.filterwarnings("ignore", message="invalid value encountered in divide")

class dataset_clip_video_audio_fmri(Dataset):
    def __init__(self, 
                data_path,
                config,
                split,
                ):

        super().__init__()

        ################################################
        ##############       CONFIG       ##############
        ################################################

        #self.task = config['data']['task']
        #sampling = config['mesh_resolution']['sampling'] ## TO remove
        ico = config['mesh_resolution']['ico_mesh']
        sub_ico = config['mesh_resolution']['ico_grid']
        #self.balance = config['data']['balance'] ## TO remove
        self.model = config['MODEL']
        self.config =config
        
        self.filedir =  data_path
        self.split = split
        #self.configuration = config['data']['configuration']
        self.dataset = config['data']['dataset']
        self.path_to_workdir = config['data']['path_to_workdir']
        #self.hemi = config['data']['hemi'] # TO remove
        self.dataset = config['data']['dataset']
        self.augmentation = config['augmentation']['prob_augmentation']
        self.normalise = config['data']['normalise']
        self.use_confounds = config['training']['use_confounds']
        self.use_cross_val = config['training']['use_cross_validation']
        self.modality = config['data']['modality']
        self.path_to_template = config['data']['path_to_template']
        self.warps_ico = config['augmentation']['warp_ico']
        self.nbr_vertices = config['ico_{}_grid'.format(sub_ico)]['num_vertices']
        self.nbr_patches = config['ico_{}_grid'.format(sub_ico)]['num_patches']
        self.masking = config['data']['masking']
        #self.demean = config['data']['demean']
        self.clip_duration =  config['clip']['clip_duration']
        #self.subset = config['data']['subset'] # TO remove
        #self.single_subject = config['data']['single_subject']
        self.path_to_data = config['data']['path_to_data']
        if 'audio' in self.model:
            self.audio_loader = config['audio_model']['audio_loader']
        if 'video' in self.model:
            self.video_loader = config['video_model']['video_loader']

        ### fmri
        self.nbr_frames_to_extract = config['fMRI']['nbr_frames']
        self.temporal_window = config['fMRI']['window']
        self.sampling_type = config['fMRI']['sampling_type']
        self.temporal_rep = config['fMRI']['temporal_rep']
        self.temporal_lag = config['fMRI']['temporal_lag']
        self.movie_sampling_strategy = config['data']['movie_sampling_strategy']
        self.video_frame_id = False

        if config['clip']['fmri_encoder'] == 'sit':
            self.patching=True
            self.channels = [0] # by default
            self.num_channels = len(self.channels)

        else:
            raiseExceptions('model not implemented yet')

        ################################################
        ##############       LABELS       ##############
        ################################################

        self.split_info = pd.read_csv('{}/labels/{}/{}.csv'.format(config['data']['path_to_workdir'],
                                                                             config['data']['train_val_split'],
                                                                                split))                        

                
        self.subject_ids = self.split_info['ids']
        self.subject_movies = self.split_info['movie']
        self.subject_sessions = self.split_info['session']
        self.subject_hemi = self.split_info['hemi']
        self.nbr_frames_movie = self.split_info['nbr_frames']


        ########################################################################
        ##############       DATA AUGMENTATION & PROCESSING       ##############
        ########################################################################
        
        self.triangle_indices = pd.read_csv('{}/patch_extraction/triangle_indices_ico_{}_sub_ico_{}.csv'.format(config['data']['path_to_workdir'],ico,sub_ico))    
        print('ico res: {}'.format(sub_ico))
        
        if self.masking and self.dataset == 'HCP': # for HCP
            if split == 'train':
                print('Masking the cut: HCP mask')
            self.mask = np.array(nb.load('{}/L.atlasroi.40k_fs_LR.shape.gii'.format(self.path_to_template)).agg_data())

        if 'video' in self.model:
            if self.video_loader == 'numpy':
                file_path = '{}/HCP/movies/Pre_20140821_version/video_embeddings/{}s/embeddings_movie_1.h5'.format(self.path_to_data,self.clip_duration)
                with h5py.File(file_path, 'r') as f:
                    self.movie_embeddings_1 = f['final']['activations'][:].squeeze()
                file_path = '{}/HCP/movies/Pre_20140821_version/video_embeddings/{}s/embeddings_movie_2.h5'.format(self.path_to_data,self.clip_duration)
                with h5py.File(file_path, 'r') as f:
                    self.movie_embeddings_2 = f['final']['activations'][:].squeeze()
                file_path = '{}/HCP/movies/Pre_20140821_version/video_embeddings/{}s/embeddings_movie_3.h5'.format(self.path_to_data,self.clip_duration)
                with h5py.File(file_path, 'r') as f:
                    self.movie_embeddings_3 = f['final']['activations'][:].squeeze()
                file_path = '{}/HCP/movies/Pre_20140821_version/video_embeddings/{}s/embeddings_movie_4.h5'.format(self.path_to_data,self.clip_duration)
                with h5py.File(file_path, 'r') as f:
                    self.movie_embeddings_4 = f['final']['activations'][:].squeeze()
        
        
        if 'audio' in self.model:
            if self.audio_loader == 'numpy':
                file_path = '{}/HCP/movies/Pre_20140821_version/audio_embeddings/{}s/embeddings_audio_movie_1.h5'.format(self.path_to_data,self.clip_duration)
                with h5py.File(file_path, 'r') as f:
                    self.audio_embeddings_1 = f['final']['activations'][:].squeeze()
                file_path = '{}/HCP/movies/Pre_20140821_version/audio_embeddings/{}s/embeddings_audio_movie_2.h5'.format(self.path_to_data,self.clip_duration)
                with h5py.File(file_path, 'r') as f:
                    self.audio_embeddings_2 = f['final']['activations'][:].squeeze()
                file_path = '{}/HCP/movies/Pre_20140821_version/audio_embeddings/{}s/embeddings_audio_movie_3.h5'.format(self.path_to_data,self.clip_duration)
                with h5py.File(file_path, 'r') as f:
                    self.audio_embeddings_3 = f['final']['activations'][:].squeeze()
                file_path = '{}/HCP/movies/Pre_20140821_version/audio_embeddings/{}s/embeddings_audio_movie_4.h5'.format(self.path_to_data,self.clip_duration)
                with h5py.File(file_path, 'r') as f:
                    self.audio_embeddings_4 = f['final']['activations'][:].squeeze()
        
        self.fmri_session_structure = {}
        self.fmri_session_structure['1'] = {}
        self.fmri_session_structure['1']['rest_period_1'] = [0,19]
        self.fmri_session_structure['1']['movie_sample_1'] = [20,263]
        self.fmri_session_structure['1']['rest_period_2'] = [264,284]
        self.fmri_session_structure['1']['movie_sample_2'] = [285,504]
        self.fmri_session_structure['1']['rest_period_3'] = [505,525]
        self.fmri_session_structure['1']['movie_sample_3'] = [526,713]
        self.fmri_session_structure['1']['rest_period_4'] = [713,733]
        self.fmri_session_structure['1']['movie_sample_4'] = [734,797]
        self.fmri_session_structure['1']['rest_period_5'] = [798,817]
        self.fmri_session_structure['1']['test_retest'] = [818,900]
        self.fmri_session_structure['1']['rest_period_6'] = [901,920]

        self.fmri_session_structure['2'] = {}
        self.fmri_session_structure['2']['rest_period_1'] = [0,19]
        self.fmri_session_structure['2']['movie_sample_1'] = [20,246]
        self.fmri_session_structure['2']['rest_period_2'] = [247,266]
        self.fmri_session_structure['2']['movie_sample_2'] = [267,525]
        self.fmri_session_structure['2']['rest_period_3'] = [526,544]
        self.fmri_session_structure['2']['movie_sample_3'] = [545,794]
        self.fmri_session_structure['2']['rest_period_4'] = [795,814]
        self.fmri_session_structure['2']['test_retest'] = [815,897]
        self.fmri_session_structure['2']['rest_period_5'] = [898,917]

        self.fmri_session_structure['3'] = {}
        self.fmri_session_structure['3']['rest_period_1'] = [0,19]
        self.fmri_session_structure['3']['movie_sample_1'] = [20,199]
        self.fmri_session_structure['3']['rest_period_2'] = [200,220]
        self.fmri_session_structure['3']['movie_sample_2'] = [221,404]
        self.fmri_session_structure['3']['rest_period_3'] = [405,424]
        self.fmri_session_structure['3']['movie_sample_3'] = [425,628]
        self.fmri_session_structure['3']['rest_period_4'] = [629,649]
        self.fmri_session_structure['3']['movie_sample_4'] = [650,791]
        self.fmri_session_structure['3']['rest_period_5'] = [792,811]
        self.fmri_session_structure['3']['test_retest'] = [812,894]
        self.fmri_session_structure['3']['rest_period_6'] = [895,914]

        self.fmri_session_structure['4'] = {}
        self.fmri_session_structure['4']['rest_period_1'] = [0,19]
        self.fmri_session_structure['4']['movie_sample_1'] = [20,252]
        self.fmri_session_structure['4']['rest_period_2'] = [253,271]
        self.fmri_session_structure['4']['movie_sample_2'] = [272,501]
        self.fmri_session_structure['4']['rest_period_3'] = [502,521]
        self.fmri_session_structure['4']['movie_sample_3'] = [522,777]
        self.fmri_session_structure['4']['rest_period_4'] = [778,797]
        self.fmri_session_structure['4']['test_retest'] = [798,880]
        self.fmri_session_structure['4']['rest_period_5'] = [881,900]
    
            
    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self,idx):
        
        #sample one of the frame at random
        row = self.split_info.loc[idx]

        #get only the selected movie
        nbr_frames = row['nbr_frames']
        movie = row['movie']

        #sample one fMRI frame
        if self.movie_sampling_strategy=='any':
            frame = random.randint(0, nbr_frames-self.nbr_frames_to_extract-1)
            #print(frame)
        elif self.movie_sampling_strategy == 'half':
            if self.split == 'train':
                #print('using half strategy')
                session_info = self.fmri_session_structure[str(movie)]
                # Training: sample from the first half of each movie segment
                # Collect all movie sample segments
                movie_samples = [session_info[key] for key in session_info if 'movie_sample' in key]

                # Choose one movie segment at random
                chosen_segment = random.choice(movie_samples)
                
                # Determine the range for sampling in the first half
                start, end = chosen_segment
                half_point = (start + end) // 2
                sample_range = (start, half_point)
                frame = random.randint(sample_range[0], sample_range[1] - self.temporal_window)
                #print('training',movie_samples, frame)
            
            elif self.split == 'val':
                session_info = self.fmri_session_structure[str(movie)]
                # Testing: sample from the second half of each movie segment
                # Collect all movie sample segments
                movie_samples = [session_info[key] for key in session_info if 'movie_sample' in key]

                # Choose one movie segment at random
                chosen_segment = random.choice(movie_samples)
                
                # Determine the range for sampling in the second half
                start, end = chosen_segment
                half_point = (start + end) // 2
                sample_range = (half_point + 1, end)
                frame = random.randint(sample_range[0], sample_range[1] - self.temporal_window)
                #print('validation',movie_samples, frame)
            else:
                raise NotImplementedError
            
        t0_delayed = max(0, frame-self.temporal_lag)
            
        frames_to_extract = self.extract_fmri_clip(nbr_frames,t_0=frame)

        input_data = []
        for f in frames_to_extract:
            input_data.append(self.get_half_hemi(idx,f.cpu().numpy()))

        input_data = np.stack(input_data,axis=0)
        #print('input data shape - {}'.format(input_data.shape))

        input_data = self.normalise_(input_data)
        
        if self.patching:
            sequence = self.get_sequence(input_data)
            if self.nbr_frames_to_extract > 1:
                sequence = self.temporal_mixing(sequence)

        ##########################################
        ############### LOAD VIDEO ###############
        ##########################################
                        
        if movie==1:
            movie_name= 'MOVIE1_CC1'
        elif movie ==2:
            movie_name = 'MOVIE2_HO1'
        elif movie ==3:
            movie_name = 'MOVIE3_CC2'
        else:
            movie_name = 'MOVIE4_HO2'
            
        clip_id = str(t0_delayed).zfill(3)
    
        if 'video' in self.model:
            if self.video_loader == 'numpy':
                if movie==1:
                    video_clip = torch.from_numpy(self.movie_embeddings_1[int(clip_id),:]).float()
                elif movie==2:
                    video_clip = torch.from_numpy(self.movie_embeddings_2[int(clip_id),:]).float()
                elif movie==3:
                    video_clip = torch.from_numpy(self.movie_embeddings_3[int(clip_id),:]).float()
                else:
                    video_clip = torch.from_numpy(self.movie_embeddings_4[int(clip_id),:]).float()
                                
            elif self.video_loader == 'video_encoder':
                video_clip = '/home/sd20/data/HCP/movies/Pre_20140821_version/clips_7T_{}/{}s/clip_{}.mp4'.format(movie_name,self.clip_duration,clip_id)
            
        if 'audio' in self.model:
            if self.audio_loader == 'numpy':

                if movie==1:
                    audio_clip = torch.from_numpy(self.audio_embeddings_1[int(clip_id),:]).float()
                elif movie==2:
                    audio_clip = torch.from_numpy(self.audio_embeddings_2[int(clip_id),:]).float()
                elif movie==3:
                    audio_clip = torch.from_numpy(self.audio_embeddings_3[int(clip_id),:]).float()
                else:
                    audio_clip = torch.from_numpy(self.audio_embeddings_4[int(clip_id),:]).float()

            elif self.audio_loader == 'audio_encoder':
                audio_clip = '/home/sd20/data/HCP/movies/Pre_20140821_version/audio_{}/{}s/audio_clip_{}.wav'.format(movie_name,self.clip_duration,clip_id)            
                audio_clip, sample_rate = torchaudio.load(audio_clip)      

        if ('audio' in self.model) and (not 'video' in self.model):
            return (torch.from_numpy(sequence).float(),torch.Tensor(audio_clip).float())
        
        elif ('video' in self.model) and (not 'audio' in self.model):
            return (torch.from_numpy(sequence).float(),video_clip)
        
        elif ('audio' in self.model) and ('video' in self.model):
            return (torch.from_numpy(sequence).float(),video_clip, audio_clip, frame, t0_delayed)
        
    def get_half_hemi(self,idx,frame):

        path = os.path.join(self.filedir,
                            str(self.subject_ids.iloc[idx]),
                            'tfMRI_MOVIE{}_7T_{}'.format(self.subject_movies[idx],self.subject_sessions[idx]),
                            'frames_ico6_{}'.format(self.subject_hemi[idx]),
                            'frame_{}.{}.shape.gii'.format(str(frame).zfill(3),self.subject_hemi[idx]))
        #print(path)
        data =  np.array(nb.load(path).agg_data())
        
        if len(data.shape)==1:
            data = np.expand_dims(data,0)

        return data.squeeze()

    def demean_(self,data):
        
        if self.demean:
            data = (data - data.mean(axis=0).reshape(1,-1))
            
        return data
    
    def normalise_(self,data):

        if self.masking:
            non_masked_vertices = self.mask>0
            if self.normalise=='sub-standardise':
                data[:,non_masked_vertices] = (data[:,non_masked_vertices] - data[:,non_masked_vertices].mean(axis=1).reshape(data.shape[0],1))/data[:,non_masked_vertices].std(axis=1).reshape(data.shape[0],1)
        
        return data

    def extract_fmri_clip(self,nbr_fames,t_0=None):

        if t_0==None:
            t_0 = torch.randint(low = 0, high= self.video_length-self.temporal_window,size=(1,)).item()

        t_0 = min(t_0, t_0+self.temporal_lag) ### why ??

        if self.sampling_type == 'uniform':
            #from T0 to T0+T uniformly
            #print(t_0 , min(t_0+self.temporal_window,nbr_fames)-1)
            frames_to_extract = torch.round(torch.linspace(t_0,min(t_0+self.temporal_window,nbr_fames)-1,self.nbr_frames_to_extract),).int()
            #print(frames_to_extract)

        elif self.sampling_type == 'chunk':
            #sampling a chunk of frames from T0+T0'
            t_0_prime = torch.randint(low = t_0, high=t_0 + self.temporal_window - self.nbr_frames_to_extract-1,size=(1,)).item()
            frames_to_extract = [t_0_prime+i for i in range(self.nbr_frames_to_extract)]

        elif self.sampling_type == 'random':
            #sampling random frames starting from T0
            frames_to_extract = torch.cat((torch.tensor([t_0]),torch.randperm(self.temporal_window-1)[:self.nbr_frames_to_extract-1].sort()[0]+min(t_0, nbr_fames-self.temporal_window)+1))
            #print(frames_to_extract)
        else:
            raise('Not implemented yet')
        
        return frames_to_extract
    
    def temporal_mixing(self,sequence):

        #print('******* temporal mixing - {} *******:'.format(self.temporal_rep))

        if self.temporal_rep == 'concat':

            #print('sequence: {}'.format(sequence.shape))
            #### Not sure which option to chose, look at smae_video_dev.ipynb
            #sequence = rearrange(sequence, 't n v -> 1 (n t) v') #concat patches
            sequence = rearrange(sequence, 't n v -> 1 (t n) v') #concat frames
            #print('sequence: {}'.format(sequence.shape))
        
        elif self.temporal_rep == 'avg':

            #print('sequence: {}'.format(sequence.shape))
            sequence = reduce(sequence, 't n v -> 1 n v', 'mean')
            #print('sequence: {}'.format(sequence.shape))

        elif self.temporal_rep == 'mix':

            #print('sequence: {}'.format(sequence.shape))
            mask = np.eye(self.nbr_frames_to_extract,dtype=bool)[:, np.random.choice(self.nbr_frames_to_extract, self.nbr_patches)]
            sequence = sequence[mask][np.newaxis,:,:]
            #print('sequence: {}'.format(sequence.shape))

        elif self.temporal_rep == 'tubelet':

            print('sequence: {}'.format(sequence.shape))
            
        elif self.temporal_rep == 'channels':

            #print('sequence: {}'.format(sequence.shape))
            #import pdb;pdb.set_trace()
            return sequence

        else:
            raise('Not implemented yet')
        
        return sequence
    
    ############ AUGMENTATION ############

    def get_sequence(self,data):

        sequence = np.zeros((self.nbr_frames_to_extract, self.nbr_patches, self.nbr_vertices))
        #print(sequence.shape)
        #print(data.shape)
        for j in range(self.nbr_patches):
            indices_to_extract = self.triangle_indices[str(j)].to_numpy()
            sequence[:,j,:] = data[:,indices_to_extract]
        return sequence

    def apply_rotation(self,data):

        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to('cpu'),device='cpu')

        rotation_angle = np.round(random.uniform(-self.max_degree_rot,self.max_degree_rot),2)
        axis = random.choice(['x','y','z'])

        r = R.from_euler(axis, rotation_angle, degrees=True)

        new_coord = np.asarray(r.apply(self.coord_ico6),dtype=np.float32)

        rotated_moving_img = bilinear_sphere_resample(torch.Tensor(new_coord),img, 100, 'cpu')

        return rotated_moving_img.numpy().T

    def apply_non_linear_warp(self,data):

        # chose one warps at random between 0 - 99

        id = np.random.randint(0,100)
        img = lat_lon_img_metrics('{}/surfaces/'.format(self.path_to_workdir),torch.Tensor(data.T).to('cpu'),device='cpu')
        warped_grid = nb.load('{}/warps/resample_ico6_ico_{}/ico_{}_{}.surf.gii'.format(self.path_to_template,self.warps_ico,self.warps_ico, id)).agg_data()
        warped_moving_img = bilinear_sphere_resample(torch.Tensor(warped_grid[0]), img, 100, 'cpu')
        return warped_moving_img.numpy().T
    
    ############ LOGGING ############

    def logging(self):
        
        if self.split == 'train':
                print('Using {} channels'.format(self.channels))
        
        if self.split == 'train':
            if self.normalise == 'sub-standardise':
                print('Normalisation: Subject-wise standardised')
            elif self.normalise == 'group-standardise':
                print('Normalisation: Group-wise standardised')
            elif self.normalise == 'normalise':
                print('Normalisation: Normalised')
            else:
                print('Normalisation: Not normalised') 

        print('')
        print('#'*30)
        print('######## Augmentation ########')
        print('#'*30)
        print('')
        if self.augmentation:
            print('Augmentation: ratio {}'.format(self.augmentation))
            if self.rotation:
                print('     - rotation with probability: {} and max abs degree {}'.format(self.rotation,self.max_degree_rot))
            else:
                print('     - rotations: no')
            if self.shuffle:
                print('     - shuffling with probability: {}'.format(self.shuffle))
            else:
                print('     - shuffling: no')
            if self.warp:
                print('     - non-linear warping with probability: {}'.format(self.warp))
            else:
                print('     - non-linear warping: no')
        else:
            print('Augmentation: NO')

