---
layout: post
title: Xavier Initialization

excerpt: The curious meaning of the Xavier weight initializer in Caffe and other deep learning frameworks.
---

From the <a href="http://caffe.berkeleyvision.org/gathered/examples/mnist.html">Caffe Berkeley Vision</a> link, you may have seen that a Convolutional layer can be defined as:

```
layer {
  name: "conv1"
  type: "Convolution"
  param { lr_mult: 1 }
  param { lr_mult: 2 }
  convolution_param {
    num_output: 20
    kernel_size: 5
    stride: 1
    weight_filler { # WEIGHT INITIALIZATION
      type: "xavier"
    }
    bias_filler {
      type: "constant"
    }
  }
  bottom: "data"
  top: "conv1"
}
```
where the weight filler uses the $$\texttt{xavier}$$ algorithm that automatically determines the scale of initialization based on the number of input and output neurons.

## What is Xavier and how is it important? 

Imagine that your weights are initially very close to 0. What happens is that the signals shrink as it goes through each layer until it becomes too tiny to be useful. Now if your weights are too big, the signals grow at each layer it passes through until it is too massive to be useful.

By using Xavier initialization, we make sure that the weights are not too small but not too big to propagate accurately the signals. From my tests, it turns out that initialization is surprisingly important. A marked difference can appear with only 3-4 layers in the network.

<a href="http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf">Glorot and Bengio (2009)</a> suggested to initialize the weights from a distribution with zero mean and variance:

$$Var(W) = \frac{2}{n_{in}+n_{out}} $$
 
where $$n_{in}$$ and $$n_{out}$$ are respectively the number of inputs and outputs of your layer.

# Proof

Suppose we have an input $$X$$ with $$n$$ components and a fully connected layer (also denoted linear or dense) with random weights $$W$$ that outputs a number $$Y$$ such that

$$Y = W_1 X_1 +W_2 X_2 + ... + W_nX_n$$

To make sure that the weights remain in a reasonable range, we expect that $$Var(Y) = Var(X_i)_{i \in [1,n]}$$

We also know how to compute the variance of the product of two random variables. Therefore

$$ Var(W_iX_i)=E[X_i]^2Var(W_i)+E[W_i]^2Var(X_i)+Var(W_i)Var(X_i)$$

Both our inputs and weights have a mean 0. It simplifies to

$$ Var(W_iX_i) = Var(W_i)Var(X_i) $$

Now we make a further assumption that the $$X_i$$ and $$W_i$$ are all independent and identically distributed (iid).

$$Var(Y) = Var(W_1X_1+W_2X_2+....+W_nX_n)=nVar(W_i)Var(X_i) $$

It turns that, if we want to have $$Var(Y) = Var(X_i)$$, we must enforce the condition $$ nVar(W_i) = 1$$.

$$ Var(W_i)= \frac{1}{n} = \frac{1}{n_{in}}$$

Still from <a href="http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf">Glorot and Bengio (2009)</a>, they found that if we go through the same steps for the backpropagated signal, you find that you need

$$ Var(W_i)= \frac{1}{n_{out}}$$

Those two constraints can only be satisfied if $$n_{in} = n_{out}$$. In practice, it restricts too much. As a compromise, they roughly took the average of the two

$$ Var(W_i)= \frac{2}{n_{in} + n_{out}}$$

