---
layout: post
title: Configuration of a GPU for Deep Learning (Theano)

excerpt: Easy tutorial on how to configure properly a GPU for Deep Learning with Ubuntu 14.04 x64 and GTX 460 (this card does not support of CuDNN).
---

I assume that you are running a freshly installed version of Ubuntu or Kubuntu 14.04 LTS x64 and that you have a NVIDIA GPU (at least GTX 460). GTX 980 and Titan X should be better :)

## Installation of theano, NVIDIA drivers and CUDA toolkit

```
# Install Theano
sudo apt-get install python-numpy python-scipy python-dev python-pip python-nose g++ libopenblas-dev git
sudo pip install Theano

# Install Nvidia drivers, CUDA and CUDA toolkit, following some instructions from http://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb # Got the link at https://developer.nvidia.com/cuda-downloads
sudo dpkg -i cuda-repo-ubuntu1404-7-5-local_7.5-18_amd64.deb
sudo apt-get update
sudo apt-get install cuda

sudo reboot
```

At that point, running ```nvidia-smi``` should work, but running ```nvcc``` won't work.

```
# Execute in console, or (add in ~/.bash_profile then run "source ~/.bash_profile"):
export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
```

At that point, both ```nvidia-smi``` and ```nvcc``` should work.

## Sanity Check to make sure Theano uses your GPU

To test whether Theano is able to use the GPU:

Copy-paste the following in gpu_test.py:

```
# Start gpu_test.py
# From http://deeplearning.net/software/theano/tutorial/using_gpu.html#using-gpu
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in xrange(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
# End gpu_test.py
```

and run it:

```
THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python gpu_test.py
```

which should return:	

```
philipperemy:~$ THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32' python gpu_test.py
Using gpu device 0: GeForce GTX 690
[GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>), HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
Looping 1000 times took 0.658292 seconds
Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
  1.62323296]
Used the gpu
```

## CuDNN installation

<i>The NVIDIA CUDA® Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks. cuDNN provides highly tuned implementations for standard routines such as forward and backward convolution, pooling, normalization, and activation layers. cuDNN is part of the NVIDIA Deep Learning SDK.</i>

<b>Installing CuDNN is VERY safe. It brings you amazing performance improvements for Neural Networks (Recurrent, Convolutional, Fully Connected...). Even if your graphic card is not compatible with this library, Theano is smart enough to disable this optimization.</b>

To add cuDNN, please follow these steps:

Download cuDNN from <a href="https://developer.nvidia.com/rdp/cudnn-download">NVIDIA CUDNN Download</a> (need registration, which is free)

```
tar -xvf cudnn-7.0-linux-x64-v3.0-prod.tgz
```

Copy the ```*.h``` files to ```CUDA_ROOT/include``` and the ```*.so*``` files to ```CUDA_ROOT/lib64``` (by default, ```CUDA_ROOT``` is ```/usr/local/cuda``` on Linux).

By default, Theano will detect if it can use cuDNN. If so, it will use it. If not, Theano optimizations will not introduce cuDNN ops. So Theano will still work if the user did not introduce them manually.

To get an error if Theano can not use cuDNN, use this Theano flag: ```optimizer_including=cudnn```

Example:

```
THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn' python gpu_test.py
```

## Running MNIST on the GPU (keras)

<i>Keras is a minimalist, highly modular neural networks library, written in Python and capable of running on top of either TensorFlow or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.</i>

Let's clone the repository

```
git clone https://github.com/fchollet/keras.git
cd keras
sudo python setup.py install
cd examples
THEANO_FLAGS='mode=FAST_RUN,device=gpu,floatX=float32,optimizer_including=cudnn' python mnist_cnn.py
```
