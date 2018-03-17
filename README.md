# OpenSpeech-Prediciton
The backend training system for the OpenSpeech app.

## SETUP

### Anaconda:

  If you wish to keep this install separate from your base python installation follow these instructions. If you are using Windows you must use Anaconda:
  
    1. Download and install Anaconda from here: https://www.continuum.io/downloads
    2. Windows, open the conda shell and run:
            create -n tensorflow python=3.5
            activate tensorflow
    3. Linux, open a terminal and run:
            conda create -n tensorflow python=3.5
            source activate tensorflow
            
After you have installed conda you can run the activate command for your system to return to your tensorflow installation. When using anaconda make sure you have activated the tensorflow environment either in the conda shell(Windows) or in a terminal(Linux) before running the rest of the setup.

### CPU only:

Run:

    pip install tensorflow -U
    pip install nltk -U
    pip install matplotlib -U
    pip install scikit-learn -U
    conda install -c scipy==0.19.0
    pip install keras -U
    pip install h5py -U

### NVIDIA GPU:
First, check if your GPU is supported here: https://developer.nvidia.com/cuda-gpus

Ensure that CUDA 8.0 and cuDNN are installed from the following locations:

  1. CUDA: https://developer.nvidia.com/cuda-downloads
  
  2. cuDNN: https://developer.nvidia.com/cudnn
  
  3. Add the location of the cuDNN files to your system path. For windows this should be:
      %EXTRACTED_LOCATION%/cuda/lib/x64
      
  4. Copy the following file into %PATH%/bin/:
      %EXTRACED_LOCATION%/cuda/bin/cudnn64_5.dll
  
  5. Copy the following file into %PATH%/include/:
      %EXTRACTED_LOCATION%/cuda/include/cudnn.h
      
  6. Copy the following file into %PATH%/bin/:
      %EXTRACTED_LOCATION%/cuda/lib/x64/cudnn.lib
      
  Where %PATH% is your Cuda install locaiton(such as "C:/Program Files/NVIDIA GPU Computing TOolkit/CUDA/v8.0/lib/x64/"
  
### Then run:

Windows:

    pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/windows/gpu/tensorflow_gpu-1.0.1-cp35-cp35m-win_amd64.whl
    pip install --upgrade nltk
    pip install --upgrade matplotlib
    pip install --upgrade scikit-learn
    conda install -c scipy==0.19.0
    pip install keras -U
    pip install h5py -U

Linux:

    pip install --ignore-installed --upgrade  https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.1-cp34-cp34m-linux_x86_64.whl
    pip install --upgrade nltk
    pip install --upgrade matplotlib
    pip install --upgrade scikit-learn
    pip install --upgrade scipy==0.19.0
    pip install keras -U
    pip install h5py -U

### Download this project:
Either download and unzip this project somewhere with a decent amount of space(I suggest at least 10GB) or run:

    git clone https://github.com/SirRujak/OpenSpeech-Prediction

### Download dataset:
Download the swiftkey dataset from:

https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip

and place it in the Datasets directory.

### Training the network:

Once your tensorflow environment is set up run "python trainer.py" from the src directory.
