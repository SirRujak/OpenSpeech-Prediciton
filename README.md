# OpenSpeech-Prediciton
The backend training system for the OpenSpeech app.

SETUP
CPU:
Anaconda:
  If you wish to keep this install separate from your base python installation follow these instructions:
    1.
    2.
Run pip install tensorflow -U

NVIDIA GPU:
Ensure that CUDA 8.0 and cuDNN are installed from the following locations:
  CUDA: http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#axzz4cvpOw4Ul
  cuDNN: https://developer.nvidia.com/cudnn
Then run: pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.0.1-cp35-cp35m-win_amd64.whl

Once your tensorflow environment is set up run "python trainer.py" from the src directory.
