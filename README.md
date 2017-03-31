# OpenSpeech-Prediciton
The backend training system for the OpenSpeech app.

## SETUP

### Anaconda:

  If you wish to keep this install separate from your base python installation follow these instructions:
    1. Download and install Anaconda from here: https://www.continuum.io/downloads
    2. Create a new conda environment by running: conda create -n tensorflow
    3. Windows, open the conda shell and run: activate tensorflow
    4. Linux, open a terminal and run: activate tensorflow

### CPU only:

Run pip install tensorflow -U

### NVIDIA GPU:

Ensure that CUDA 8.0 and cuDNN are installed from the following locations:

  1. CUDA: http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#axzz4cvpOw4Ul
  
  2. cuDNN: https://developer.nvidia.com/cudnn
  
Then run:

pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.0.1-cp35-cp35m-win_amd64.whl

### Training the network:

Once your tensorflow environment is set up run "python trainer.py" from the src directory.
