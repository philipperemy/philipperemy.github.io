---
layout: post
title: Deep Learning (Machine Learning) applied to Tinder

excerpt: We will look at Convolutional Neural Networks, with a fun example of training them to automatically swipe left/right based on a scraped dataset of 200 thousand of pictures from Instagram.
---

![png](/images/tinder/tinder-1.png){: .center-image }

Hey guys! Today we're going to see how to apply Deep Learning to Tinder in order to make your own bot able to swipe either left/right automatically. More specifically, we're going to use Convolutional Neural Networks. Never heard of them? Those models are great: they recognize objects, places and people in your personal photos, signs, people and lights in self-driving cars, crops, forests and traffic in aerial imagery, various anomalies in medical images and all kinds of other useful things. But once in a while these powerful visual recognition models can also be warped for distraction, fun and amusement. In this experiment, we're going to do that:

- We'll take a a powerful, 5-million-parameter almost state-of-the-art Convolutional Neural Network, feed it thousands of images scraped from the internet, and train it to classify between attractive pictures from less attractive ones. 
- The dataset is composed of 151k images, scraped from Instagram and Tinder (50-50). Because we don't have access to the full Tinder database to calculate the attractiveness ratio (how many right swipes over the total number of views), we augmented the dataset with pictures from Instagram for which we know the attractiveness is high (clue: Kim Kardashian instagram). 

For those who are curious about what's going on behind the scenes, I highly recommend this course from Stanford University that you can find here [http://cs231n.github.io/convolutional-networks/](http://cs231n.github.io/convolutional-networks/).

## Data

All pictures from Instagram are tagged LIKE and pictures from Tinder are tagged NOPE. We will see later on how this split can be useful in our Tinder bot. Here is what it looks like:

![png](/images/tinder/5.png){: .center-image }
![png](/images/tinder/6.png){: .center-image }
![png](/images/tinder/7.png){: .center-image }
![png](/images/tinder/8.png){: .center-image }

Okay I don't get it. Let's say we have a perfect model with 100% accuracy. We feed some random pictures of Tinder. It's going to be classified as NOPE all the time according to how the dataset is defined?

The answer is a partial yes. Here are some examples of what an imperfect model can output:

![png](/images/tinder/11.png){: .center-image }


![png](/images/tinder/12.png){: .center-image }

It translates in the fact that not only the model can predict the class (LIKE or NOPE) but also it can give a confidence percentage. For the second picture, the LIKE conviction reaches 99.84% while it tops at 96.46% for the first picture. We can make the conclusion that the model is less sure (to some extent) for the first picture. Empirically, the model will always output values with a very high confidence (either close to 100 or close to 0). It can lead to a wrong analysis if not taken seriously. The trick here is to specify a low threshold, say 40% slighly lower than the default 50%, for which all photos above this limit will be categorized as LIKE. This also increases the number of times the model will output a LIKE value from a Tinder picture (If we don't do that, we only rely on True Negatives).

## Bot algorithm

A profile usually consists in a combination of more than one picture. We consider the algorithm that if at least one picture has the status LIKE, we swipe right. If all the pictures are marked as NOPE by the Conv Net, we swipe left. We don't make any analysis based on the descriptions and/or age. The whole bot can swipe several times per second, more than any human could do.

Applying some state-of-the-art NLP techniques on the descriptions would definitely be a very good way of improvement. One's does not only swipe on the pictures. Do they?

Below is an execution of a single run:

![png](/images/tinder/13.png){: .center-image }

With the images, we get:

![png](/images/tinder/run_0.jpg){: .center-image }
![png](/images/tinder/run_1.jpg){: .center-image }
![png](/images/tinder/run_2.jpg){: .center-image }
![png](/images/tinder/run_3.jpg){: .center-image }
![png](/images/tinder/run_4.jpg){: .center-image }



## Behind the scenes (more technical)

We use the following parameters to build our dataset:

- **Image Size**: 256x256
- **Image Type**: GRAYSCALE
- **Training set**: 128422 images (85%)
- **Validation set**: 22659 images (15%)
- **Total**: 151081 images (100%)

The training set corresponds to the pictures the model will train on. The validation set consists of unseen pictures and is used to benchmark how good our model is. A usual good pattern is to increase the accuracy on both the training and the validation set at the same time. If the training accuracy goes up and the validation accuracy dramatically goes down, it means our model overfits and shows difficulties to generalize on unseen data. If the training accuracy is too slow, the model cannot learn. Two possibilities arise. Either it's impossible to distinguish between LIKE and NOPE. Or because the model is too simple to understand this difficult classification.

We use the following setup for our fun experiment:

- Core i5 6500 (6M cache, up to 3.60Ghz, 4 cores)
- GTX 1070 with 8GB of memory
- 32GB of memory

[DIGITS](https://github.com/NVIDIA/DIGITS) is a webapp for training deep learning models. It uses the framework Caffe as a backend to train Convolutional Neural Networks (Conv Nets).

![png](/images/tinder/10.png){: .center-image }

If you want to set it up yourself, I strongly advise you to:

- Pick up a graphic card with at least 6GB of memory (our experiment requires around 4GB)
- Use a fresh install of Ubuntu 14.04, 16.04
- Build DIGITS and all your frameworks from the sources [BuildDigits](https://github.com/NVIDIA/DIGITS/blob/master/docs/BuildDigits.md)

To install your NVIDIA Driver, CUDA and CuDNN, go on [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) and download the latest compatible drivers for your graphic card. Use the following settings:

- **Operating System**: Linux
- **Architecture**: x86_64
- **Distribution**: Linux
- **Version**: Up to you (16.04 or 14.04)
- **Installer Type**: SELECT **deb (local)**.

After that, run the instructions:

```
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
sudo reboot

# Execute in console, or (add in ~/.bash_profile then run "source ~/.bash_profile"), FOR nvcc:.
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
```

where you change the *deb* filename to yours. For the installation of CuDNN (not required but speeds up the training), I advice you to check my page, section **CuDNN installation** [Configuration of a GPU for Deep Learning]({% post_url 2016-04-28-configure_gpu_for_deep_learning %}). I did it with an old GTX 460 but it does not matter for CuDNN.

You will see something like this when it's properly configured:

![png](/images/tinder/3.png){: .center-image }
<center>Hardware monitoring with <b>nvidia-smi</b></center>

What we discover during the training is that the CPU is hardly used (100~150% on 400%) compared to the GPU (98~100% all the time). The process uses 4GB of memory on the GPU (50%). 


For this fun experiment, we use the Google LeNet (2014) with 22 layers, described in this paper: [https://arxiv.org/abs/1409.4842](https://arxiv.org/abs/1409.4842)

The training is pretty long: 10 hours. The convergence is almost reached at 25 epochs.

We use the following parameters to train the GoogleLeNet:

- **Data Transformations**: Subtract Mean of Image
- **Batch Size**: 32
- **Crop Size**: 224 (from 256x256 to 224x224, random crop)
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Base Learning Rate**: 0.01
- **Learning Rate Decay Policy**: Step Down (Step size 33%, Gamma 0.1)


https://github.com/kjw0612/awesome-deep-vision


![png](/images/tinder/2.png){: .center-image }

![png](/images/tinder/3.png){: .center-image }
![png](/images/tinder/4.png){: .center-image }