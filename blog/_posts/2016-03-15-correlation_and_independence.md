---
layout: post
title: Correlation and Independence

excerpt: Independence always implies zero correlation but the reverse might not be true. Here is a very simple example to illustrate it in R.
---

## If X and Y are correlated to a third random variable Z, does it mean that X and Y are also correlated? Answer is no.

<div class="imgcap">
<img src="/assets/independence_and_correlation/1.png" style="border:none; display: block; margin: 0 auto;">
<div class="thecap" style="text-align:center">Correlation plots.</div>
</div>

\\[ X \sim \mathcal{N}(0,1), Y \sim \mathcal{N}(0,1), Z = X + Y + \epsilon \text{, where } \epsilon \sim \mathcal{N}(0,1)\\]
{% highlight ruby %}
N <- 10000
X <- rnorm(N)
Y <- rnorm(N)
Z <- X + Y + rnorm(N)


cor(X,Z) # 0.57
cor(Y,Z) # 0.58
cor(X,Y) # 0.00
{% endhighlight %}

## Can dependent variables be zero correlated? Answer is yes.
\begin{align} U & \sim \mathcal{N}(0,1) \\\\ cov(U, U^2) & = E[U^3] = 0\end{align}

by definition of the odd central moments of the Normal Distribution. Therefore, $$U$$ and $$U^2$$ are strictly uncorrelated but they remain dependent through the function $$f(X) = X^2$$. 

{% highlight ruby %}
U <- rnorm(10000)
U2 <- U^2
cor(U2,U) # 0 because the third (even) moment of the normal distribution is 0.
#However U and U^2 are dependent.
{% endhighlight %}
