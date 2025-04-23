from mmaction2.mmaction.apis import inference_recognizer, init_recognizer
import matplotlib.pyplot as plt
import numpy as np

from mmengine.dataset import Compose
import os

import h5py
import argparse

import torch
import torchaudio
import numpy as np
import h5py


def extract_embeddings_last_layer(args):

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

    print("Sample Rate:", bundle.sample_rate)

    print("Labels:", bundle.get_labels())

    device = args.device

    model = bundle.get_model().to(device)

    print(model.__class__)

    list_movies = [ 'MOVIE1_CC1','MOVIE2_HO1','MOVIE3_CC2','MOVIE4_HO2']

    clip_duration = args.clip_duration

    for i, movie in enumerate(list_movies):

        print('')
        print(' ******* {} *********'.format(movie))

        # Directory containing the .wav files
        input_dir = '/home/sd20/data/HCP/movies/Pre_20140821_version/audio_{}/{}s'.format(movie,clip_duration)
        # List to store loaded audio segments
        segments = []
        # Iterate through the .wav files in the directory
        for filename in os.listdir(input_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(input_dir, filename)
                
                # Load the audio file
                waveform, sample_rate = torchaudio.load(file_path)
                
                # Append the loaded audio segment and sample rate to the lists
                segments.append(waveform)
        print(len(segments))

        activations_audio_clip = []

        with h5py.File('../outputs/embeddings/wave2vec2/{}s/embeddings_audio_movie_{}.h5'.format(clip_duration,i+1), 'w') as file:

            layer_group = file.create_group('final')

            for idx, segment in enumerate(segments):

                print('*** clip {} ***'.format(idx))

                with torch.inference_mode():
                    features, _ = model.extract_features(segments[idx].to(device))
                
                #print(features[-1].shape)

                activations_audio_clip.append(features[-1].cpu().numpy())

            activations_audio_clip = np.asarray(activations_audio_clip)

            activations_audio_clip = np.mean(activations_audio_clip,axis=2).squeeze() ## mean along the sequence/patch dimension

            # Create a dataset to store embeddings
            dataset = layer_group.create_dataset('activations', data=activations_audio_clip)
            # Store metadata as attributes
            dataset.attrs['shape'] = activations_audio_clip.shape
            dataset.attrs['dtype'] = activations_audio_clip.dtype.str

            print(activations_audio_clip.shape)


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