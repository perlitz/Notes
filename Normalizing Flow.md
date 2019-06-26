# Normalizing Flow

## Motivation - Variational Inference

The goal of variational inference is the ability to extract underlying structure of unknown data, this structure may be represented via a learned probability distribution $p_\theta(x)$. A way to go about learning $p_\theta(x)$ is by marginalizing it over a set of latent variables $z$ hopefully representing a leaner representation of the data as the coordinates of it's underlying manifold:
$$
\ln p_\theta (x)=\ln \int dz \  p_\theta (x|z)p(z).
$$
This makes a new objective of finding $p_\theta (x|z) $, this however is an intractable operation since $z$'s (latent) space might by huge and this integral cannot be solved ([Check]()). One solution to this problem is to preform *Variational inference* in which an approximate distribution is learned for the intractable distribution $p_\theta (x|z)$, such distributions are usually some tractable distributions such as normal separable ones $q_\phi (x|z)$, this way, the new objective is finding $\phi$ that makes $q_\phi (x|z)$ as close as possible to $p_\theta (x|z)$.

A common way to evaluate the similarity of distributions is the Kullback-Leibler divergence:
$$
\begin{align}
D_{KL}[q_\phi(z|x)||p_\theta(z|x)]
  =&\int dx \ q_\phi(z|x) \log\left[\frac{q_\phi(z|x)}{p_\theta(z|x)} \right]
\\=&\int dx \ q_\phi(z|x) \log\left[\frac{q_\phi(z|x)}{p_\theta(x|z)p(z)}p(x) \right]
\\=&\ \log\left[p(x) \right]\int dx \ q_\phi(z|x) -\int dx \ q_\phi(z|x) \log\left[p_\theta(x|z)\right]+\int dx \ q_\phi(z|x) \log\left[\frac{q_\phi(z|x)}{p(z)}\right]
\\=&\ \underbrace{\log p_\theta(x)}_\text{evidence}-\underbrace{\left[\ \mathbb{E}_{z\sim q(z|x)}[\log p(x|z)]-D_{KL}[q_\phi(z|x)||p(z)]\ \right]}_\text{ELBO - evidence lower bound}
\end{align}
$$
Arranging it:
$$
\begin{align}
\underbrace{\log p_\theta(x)}_\text{evidence}
= \underbrace{\left[\ \mathbb{E}_{z\sim q(z|x)}[\log p(x|z)]-D_{KL}[q_\phi(z|x)||p(z)]\ \right]}_\text{ELBO - evidence lower bound}+\underbrace{D_{KL}[q_\phi(z|x)||p_\theta(z|x)]}_{\ge0}
\end{align}
$$
The rightmost term is still intractable but knowing that  $D_{KL}[\ ]\ge0$ always, we take it that the ELBO the (the minus of the) 1st term on the right gives a lower bound on the evidence and thus a common practice is to maximize it by choosing some prior distribution $p(z)$ (say a normal one), and building a *variational autoencoder* in which the encoder network gives $z\sim q_\phi(z|x)$ from the data (this gives the expectation over the latent space) and the decoder network gives $x\sim p(x|z)$ for each $z$.
$$
\text{ELBO}=\underbrace{\mathbb{E}_{z\sim q(z|x)}[\log p(x|z)]}_\text{reconstruction term}-\underbrace{D_{KL}[q_\phi(z|x)||p(z)]}_{\text{make }q_\phi(z|x) \text{ close to }p(z)}
$$
Using this construction one can both:

1. Preform MLE on the output $p_\theta (x|z)$ as in a regular autoencoder and
2. Minimize the KL divergence of $q_\phi(z|x)$ and $p(z)$, both gaussian with the latter further chosen as prior to be a unit, zero mean one.

and get as good an approximation to $p_\theta(x|z)$ which comes in handy since sampling from $p(z)$ is as simple as sampling from a unit gaussian.

But what if we want the ELBO to be an even better bound? this requires addressing the $D_{KL}[q_\phi(z|x)||p_\theta(z|x)]$ term by minimizing it. This is where *Normalizing flows* enters the picture, it enables us to learn $q_\phi(z|x)$ by estimating the probability density of $p_\theta(z|x) $  and consequentially minimize  $D_{KL}[q_\phi(z|x)||p_\theta(z|x)]$.