# About
U-Net cycleGAN implementation in Tensorflow 2.0 for overcoming domain shift. 
A brief overview of the directories:

## debiasmedimg
An installable python module (pip install --user --editable ./debiasmedimg), which entails U-Net cycleGAN.

## data
A directory that can be accessed via `debiasmedimg.settings.DATA_DIR`. Could be used for staring data or a MongoDB database. 
Contains an example for the required data structure.

## docker_context
Contains the Dockerfile and the required files for the Docker image. 
Requirements-long contains all required modules including versions. 

## output
A directory that can be accessed via `debiasmedimg.settings.OUTPUT_DIR`. 
Tensorflow checkpoints, generated images, plots showing the training progress, etc. get saved here.

## scripts
Scripts using sacred to track experiments. An example of using U-net cycleGAN is located here.
