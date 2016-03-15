---
layout: post
title: Correlation and Independence

excerpt: Independence always implies zero correlation but the reverse might not be true. Here is a very simple example to illustrate it in R.
---

## If X and Y are correlated to a third random variable Z, does it mean that X and Y are also correlated? Answer is no.

<div class="imgcap">
<img src="/assets/independence_and_correlation/1.png" style="border:none; display: block; margin: 0 auto;">
<div class="thecap" style="text-align:center">This is a caption.</div>
</div>

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

{% highlight ruby %}
U <- rnorm(10000)
U2 <- U^2
cor(U2,U) # 0 because the third (even) moment of the normal distribution is 0.
#However U and U^2 are dependent.
{% endhighlight %}
