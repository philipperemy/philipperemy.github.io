---
layout: post
title: Bayesian Optimization Example

excerpt: Easy tutorial on how to configure properly a GPU for Deep Learning with Ubuntu 14.04 x64 and GTX 460 (this card does not support of CuDNN).
---

In this article, I will present a very useful technique to find the global maxima of any function $$f : \mathbb{R}^n \rightarrow \mathbb{R}$$. It is called Bayesian Optimization.

<i>
<b>Bayesian Optimization</b> is a constrained global optimization package built upon bayesian inference and gaussian process, that attempts to find the maximum value of an unknown function in as few iterations as possible. This technique is particularly suited for optimization of high cost functions, situations where the balance between exploration and exploitation is important.
</i>

Let us start by installing the required packages.

```
pip install numpy scipy scikit-learn bayesian-optimization
```

From there, lets start an ipython notebook and go through the example step by step.

```python
from bayes_opt import BayesianOptimization
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec
%matplotlib inline
```

# Target Function

Lets create a target 1-D function with multiple local maxima $$f(x) = e^{-(x - 2)^2} + e^{-\frac{(x - 6)^2}{10}} + \frac{1}{x^2 + 1}, $$ its maximum is at $$x = 2$$ and we will restrict the interval of interest to $$x \in (-2, 10)$$.


```python
def target(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)
```

```python
KAPPA = 5
x = np.linspace(-2, 10, 1000)
y = target(x)

plt.plot(x, y)
```

![png](visualization_files/visualization_4_1.png){: .center-image }

# Create a BayesianOptimization Object

Enter the target function to be maximized, its variable(s) and their corresponding ranges - `{'x': (-4, 4), 'y': (-3, 3)}` for a multi-variable case. A minimum number of 2 initial guesses is necessary to kick start the algorithms, these can either be random or user defined.


```python
bo = BayesianOptimization(target, {'x': (-2, 10)})
```

In this example we will use the Upper Confidence Bound (UCB) as our utility function. It has the free parameter
$$\kappa$$ which control the balance between exploration and exploitation; we will set $$\kappa=5$$ which, in this case, makes the algorithm quite bold. Additionally we will use the cubic correlation in our Gaussian Process.


```python
gp_params = {'corr': 'cubic'}
bo.maximize(init_points=2, n_iter=0, acq='ucb', kappa=KAPPA, **gp_params)
```

    [31mInitialization[0m
    [94m-----------------------------------------[0m
     Step |   Time |      Value |         x | 
        1 | 00m00s | [35m   0.98623[0m | [32m   0.3886[0m | 
        2 | 00m00s |    0.35112 |    9.2861 | 
    [31mBayesian Optimization[0m
    [94m-----------------------------------------[0m
     Step |   Time |      Value |         x | 



```python

```

# Plotting and visualizing the algorithm at each step

### Lets first define a couple functions to make plotting easier


```python
def posterior(bo, xmin=-2, xmax=10):
    xmin, xmax = -2, 10
    bo.gp.fit(bo.X, bo.Y)
    mu, sigma2 = bo.gp.predict(np.linspace(xmin, xmax, 1000).reshape(-1, 1), eval_MSE=True)
    return mu, np.sqrt(sigma2)

def plot_gp(bo, x, y):
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Gaussian Process and Utility Function After {} Steps'.format(len(bo.X)), fontdict={'size':30})
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])
    
    mu, sigma = posterior(bo)
    axis.plot(x, y, linewidth=3, label='Target')
    axis.plot(bo.X.flatten(), bo.Y, 'D', markersize=8, label=u'Observations', color='r')
    axis.plot(x, mu, '--', color='k', label='Prediction')

    axis.fill(np.concatenate([x, x[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
        alpha=.6, fc='c', ec='None', label='95% confidence interval')
    
    axis.set_xlim((-2, 10))
    axis.set_ylim((None, None))
    axis.set_ylabel('f(x)', fontdict={'size':20})
    axis.set_xlabel('x', fontdict={'size':20})
    
    utility = bo.util.utility(x.reshape((-1, 1)), bo.gp, 0)
    acq.plot(x, utility, label='Utility Function', color='purple')
    acq.plot(x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
             label=u'Next Best Guess', markerfacecolor='gold', markeredgecolor='k', markeredgewidth=1)
    acq.set_xlim((-2, 10))
    acq.set_ylim((0, np.max(utility) + 0.5))
    acq.set_ylabel('Utility', fontdict={'size':20})
    acq.set_xlabel('x', fontdict={'size':20})
    
    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
```

### Two random points

After we probe two points at random, we can fit a Gaussian Process and start the bayesian optimization procedure. Two points should give us a uneventful posterior with the uncertainty growing as we go further from the observations.


```python
plot_gp(bo, x, y)
```


![png](visualization_files/visualization_15_0.png)


### After one step of GP (and two random points)


```python
bo.maximize(init_points=0, n_iter=1, kappa=KAPPA)
plot_gp(bo, x, y)
```

    [31mBayesian Optimization[0m
    [94m-----------------------------------------[0m
     Step |   Time |      Value |         x | 
        3 | 00m01s | [35m   1.24680[0m | [32m   2.4441[0m | 



![png](visualization_files/visualization_17_1.png)


### After two steps of GP (and two random points)


```python
bo.maximize(init_points=0, n_iter=1, kappa=KAPPA)
plot_gp(bo, x, y)
```

    [31mBayesian Optimization[0m
    [94m-----------------------------------------[0m
     Step |   Time |      Value |         x | 
        4 | 00m01s |    0.71095 |    3.6300 | 



![png](visualization_files/visualization_19_1.png)


### After three steps of GP (and two random points)


```python
bo.maximize(init_points=0, n_iter=1, kappa=KAPPA)
plot_gp(bo, x, y)
```

    [31mBayesian Optimization[0m
    [94m-----------------------------------------[0m
     Step |   Time |      Value |         x | 
        5 | 00m01s | [35m   1.35597[0m | [32m   1.7659[0m | 



![png](visualization_files/visualization_21_1.png)


### After four steps of GP (and two random points)


```python
bo.maximize(init_points=0, n_iter=1, kappa=KAPPA)
plot_gp(bo, x, y)
```

    [31mBayesian Optimization[0m
    [94m-----------------------------------------[0m
     Step |   Time |      Value |         x | 
        6 | 00m01s |    0.99869 |    6.4975 | 



![png](visualization_files/visualization_23_1.png)



```python

```

### After five steps of GP (and two random points)


```python
bo.maximize(init_points=0, n_iter=1, kappa=KAPPA)
plot_gp(bo, x, y)
```

    [31mBayesian Optimization[0m
    [94m-----------------------------------------[0m
     Step |   Time |      Value |         x | 
        7 | 00m01s |    0.20166 |   -2.0000 | 



![png](visualization_files/visualization_26_1.png)


### After six steps of GP (and two random points)


```python
bo.maximize(init_points=0, n_iter=1, kappa=KAPPA)
plot_gp(bo, x, y)
```

    [31mBayesian Optimization[0m
    [94m-----------------------------------------[0m
     Step |   Time |      Value |         x | 
        8 | 00m01s |    1.20559 |    1.4791 | 



![png](visualization_files/visualization_28_1.png)


### After seven steps of GP (and two random points)


```python
bo.maximize(init_points=0, n_iter=1, kappa=KAPPA)
plot_gp(bo, x, y)
```

    [31mBayesian Optimization[0m
    [94m-----------------------------------------[0m
     Step |   Time |      Value |         x | 
        9 | 00m01s | [35m   1.40136[0m | [32m   1.9759[0m | 



![png](visualization_files/visualization_30_1.png)


# Stopping

After just a few points the algorithm was able to get pretty close to the true maximum. It is important to notice that the trade off between exploration (exploring the parameter space) and exploitation (probing points near the current known maximum) is fundamental to a succesful bayesian optimization procedure. The utility function being used here (Upper Confidence Bound - UCB) has a free parameter $$\kappa$$ that allows the user to make the algorithm more or less conservative. Additionally, a the larger the initial set of random points explored, the less likely the algorithm is to get stuck in local minima due to being too conservative.


# References

- [http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf](http://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)

- [http://arxiv.org/pdf/1012.2599v1.pdf](http://arxiv.org/pdf/1012.2599v1.pdf)
