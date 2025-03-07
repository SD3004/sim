# SIM: Surface-based fMRI Analysis for Inter-Subject Multimodal Decoding from Movie-Watching Experiments

This repo contains the code for **SIM: Surface-based fMRI Analysis for Inter-Subject Multimodal Decoding from Movie-Watching Experiments**

<details>
    <summary><b> V 0.1 - 07.05.25</b></summary>
    Initial commits
    <ul type="circle">
        <li> Adding basis of the SIM codebase for tri-modal alignment</li>
    </ul>
</details>


# Installation & Set-up

## 1. Conda installation

For cuda usage and python dependencies installation please follow instructions in [install.md](docs/install.md).

# Training

The training commands are run using `torchrun` and using config files located under `/config/`.

## fMRI & Video & Audio

For training the SIM pipeline with all three modalities, please run:

```
cd tools/
torchrun --nproc_per_node=1 --nnodes=1  train_fmri_clip_ddp.py ../config/CLIP-fmri-video-audio/hparams.yml 
```

## fMRI & Video

For training the SIM pipeline with fMRI and video modalities, please run:

```
cd tools/
torchrun --nproc_per_node=1 --nnodes=1  train_fmri_clip_ddp.py ../config/CLIP-video/hparams.yml 
```

## fMRI & Video & Audio

For training the SIM pipeline with fMRI and audio modalities, please run:

```
cd tools/
torchrun --nproc_per_node=1 --nnodes=1  train_fmri_clip_ddp.py ../config/CLIP-audio/hparams.yml 
```