from mmaction2.mmaction.apis import inference_recognizer, init_recognizer
import matplotlib.pyplot as plt
import numpy as np

from mmengine.dataset import Compose
import os

import h5py
import argparse


def extract_embeddings_last_layer(args):

    config_path = './mmaction2/configs/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400-custom.py'
    checkpoint_path = 'https://download.openmmlab.com/mmaction/v1.0/recognition/videomae/vit-base-p16_videomae-k400-pre_16x4x1_kinetics-400_20221013-860a3cd3.pth'

    list_movies = [ 'MOVIE1_CC1','MOVIE2_HO1','MOVIE3_CC2','MOVIE4_HO2']

    clip_duration = args.clip_duration
    device = args.device

    for i, movie in enumerate(list_movies):
        
        print('')
        print(' ******* {} *********'.format(movie))

        with h5py.File('../outputs/embeddings/videomae/{}s/embeddings_movie_{}.h5'.format(clip_duration,i+1), 'w') as file:

            layer_group = file.create_group('final')

            model = init_recognizer(config_path, checkpoint_path, device="cuda:{}".format(device))  # device can be 'cuda:0'

            custom_cfg = model.cfg.test_pipeline

            # Define a hook to capture activations
            activations = {}
            def hook_fn(module, input, output):
                activations['value'] = output

            #model.backbone.conv5.bn.register_forward_hook(hook_fn)
            model.backbone.fc_norm.register_forward_hook(hook_fn)

            activations_movie= []

            list_clips = os.listdir('/home/sd20/data/HCP/movies/Pre_20140821_version/clips_7T_{}/{}s/'.format(movie,clip_duration))
            list_clips.sort()

            for clip in list_clips:
                
                print(clip)
                
                result, data, data1, data2 = inference_recognizer(model, '/home/sd20/data/HCP/movies/Pre_20140821_version/clips_7T_{}/{}s/'.format(movie,clip_duration)+clip, test_pipeline=Compose(custom_cfg))

                activations_movie.append(activations['value'].squeeze().cpu().numpy())
                
            activations_movie = np.asarray(activations_movie)
            print(activations_movie.shape)
            
            import pdb;pdb.set_trace()

            # Create a dataset to store embeddings
            dataset = layer_group.create_dataset('activations', data=activations_movie)

            # Store metadata as attributes
            dataset.attrs['shape'] = activations_movie.shape
            dataset.attrs['dtype'] = activations_movie.dtype.str


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Greet a user based on their name and age.")
    
    # Add arguments to the parser
    parser.add_argument("--clip-duration", type=int, required=True, help="Name of the user")
    parser.add_argument("--device", type=int, required=True, help="Name of the user")
    parser.add_argument("--last", action="store_true", help="Age of the user")

    # Parse the arguments
    args = parser.parse_args()
    print(args)

    if args.last:
        print('process last layer')
        extract_embeddings_last_layer(args)
