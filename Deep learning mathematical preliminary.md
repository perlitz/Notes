# Deep learning mathematical preliminaries



[TOC]



## Probability bits

### Probability definitions



#### Maximum likelihood Vs Cross enthropy

Maximum likelihood and cross entropy loss are usually the same thing.  The demand for maximum likelihood becomes the same as that for maximizing the cross entropy, as an example, this is the binary cross enthropy:
$$
BCE(y,x,\theta)=-\sum_{i=1}^{n} y_i \log (p_\theta(y|x_i)) + (1-y_i) \log (1-p_\theta(y|x_i))
$$
Now, for the ML, the correct likelihood for the case of binary classification is the Bernoulli $p(y|\pi)=\Pi_{i=1}^n	\pi_i^{y_i}(1-\pi_i)^{1-y_i} $. When we train the model, we wish to approximate $\pi_i$, we will represent it using $p_\theta(y|x)$, and the likelihood will be:
$$
p(y|x,\theta)=\Pi_{i=1}^n	p_\theta(y|x_i)^{y_i}(1-p_\theta(y|x_i))^{1-y_i}
$$
 And taking the log of this guy will give us:
$$
\mathcal{L}(\theta;x,y)=\sum_{i=1}^n\left[ y_i\log(p_\theta(y|x_i))+(1-y_i)\log(1-p_\theta(y|x_i) \right]
$$
Which is the same thing.

The key point behind both of these conditions is that they demand a good correspondence between the output of the learner and the real distribution, the difference is just that in ML the estimator (ML) is defined along with the likelihood (binary Bernoulli likelihood).

The same thing goes for liner regression, one can just use an $L^2$ loss or one can define a ML estimator, and a gaussian likelihood and get the same loss.

## Representation learning

[Great post by Chris Olah](<https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/>)

### Manifolds Definitions:

***Homeomorphism*** -  A transformation which preserves topological properties, A combination of homeomorphic transformations is itself a homeomorphism. Functions which are **continuous and have continuous inverses are homeomorphic**. A *Homeomorphism* which also preserves distance is called an **isometry** (a bijection which preserves distances).

***Bijection*** -  A transformation which is **one-to-one** and **onto**, a transformation which is **one-to-one** but is not **onto** is called ***Injection***, a transformation which is **onto** but is not **one-to-one** is called **surjection**

![1556106066133](C:\Users\yotampe\AppData\Roaming\Typora\typora-user-images\1556106066133.png)

***Ambient isotopy*** a Homotopy (A continuous transformation from one function to another), from my understanding it is continuous function which maps between two manifolds. *definition not done*. If a networks layer has W which is not singular and there is more than one layer, a network's representation and it's input have between them an Ambient isotopy.

 Links and knots (they are embedding of a circle in $S^1$) are 1-dimensional manifolds, but we need 4 dimensions to be able to untangle all of them. Similarly, one can need yet higher dimensional space to be able to unknot n-dimensional manifolds. All n-dimensional manifolds can be untangled in 2n+2 dimensions.

## Backpropagation

### Automatic differentiation

Backpropagation is Automatic differentiation (AD) in reverse accumulation mode

Backpropagation is a **dynamic programming** method in which calculations use saved previous ones, this method uses the adjoint as the saved variable. 

AD is not a symbolic differentiation method, the difference lies on the idea that symbolic differentiation requires a closed form while in AD can be done in parts, where each part is evaluated separately, a thing which allows for accumulation of values such as in reverse mode which is an example of dynamic programming. In AD we apply symbolic differentiation at the elementary operation level and keep intermediate numerical results.

![1556615683200](C:\Users\yotampe\AppData\Roaming\Typora\typora-user-images\1556615683200.png)

A layer's adjoint is it's derivative with respects to the output, it represents the sensitivity of the output for changes in the layer.

### Lagrange multipliers in Backpropagation

[A Great blog post](<https://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/>)

It turns out that the de facto method for handling constraints, the method Lagrange multipliers, recovers *exactly* the adjoints (intermediate derivatives) in the backprop algorithm! , [the method of Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier), is a method with which we converts a *constrained* optimization problem into an *unconstrained* one.

The idea here is that in the same manner as in backpropagation, we want to minimize some end result with respects to some initial value and want to know what one has to do with all the constrains (calculations) on the way, an example can be seen here upon taking a constrained optimization problem such as:
$$
\begin{align*}
\underset{x}{\text{argmax}}\ & f \\
\text{s.t.} \quad
a &= \exp(x) \\
b &= a^2     \\
c &= a + b   \\
d &= \exp(c) \\
e &= \sin(c) \\
f &= d + e
\end{align*}
$$
the way to attack this is by first defining the problem in the language of optimization parameters:
$$
\begin{align*}
  & \underset{\boldsymbol{x}}{\text{argmax}}\ z_n & \\
  & \text{s.t.}\quad z_i = x_i                          &\text{ for $1 \le i \le d$} \\
  & \phantom{\text{s.t.}}\quad z_i = f_i(z_{\alpha(i)}) &\text{ for $d < i \le n$} \\
  \end{align*}
$$
the first constraint is just a way to introduce the input, the second one keeps the intermediate evaluations. Input variables are the $x_1,...,x_d$ and the output in $x_n$. $α(i)α(i)$ is the list of *incoming* edges to node $i$ and $β(j)={i:j∈α(i)}$ is the set of *outgoing* edges

In order to solve the above constraint optimization problem is the Lagrange multipliers  which converts a *constrained* optimization problem into an *unconstrained* one, we *pay* with the introduction of $\boldsymbol{\lambda}$ (one per $x_i$ constraint). In this method, the main object is the Lagrangian;  
$$
\mathcal{L}\left(\boldsymbol{x}, \boldsymbol{z}, \boldsymbol{\lambda}\right)
= z_n - \sum_{i=1}^n \lambda_i \cdot \left( z_i - f_i(z_{\alpha(i)}) \right).
$$
the problem becomes that of optimizing the above:
$$
\nabla \mathcal{L}\left(\boldsymbol{x}, \boldsymbol{z}, \boldsymbol{\lambda}\right) = 0
$$
This optimization has two parts:

1. **Intermediate variables** ($\boldsymbol{z}$): Optimizing the multipliers—i.e., setting the gradient of Lagrangian w.r.t. $\boldsymbol{\lambda}$ to zero—ensures that the constraints on intermediate variables are satisfied.
   $$
   \begin{eqnarray*}
   \nabla_{\! \lambda_i} \mathcal{L}
   = z_i - f_i(z_{\alpha(i)}) = 0
   \quad\Leftrightarrow\quad z_i = f_i(z_{\alpha(i)})
   \end{eqnarray*}
   $$
     We use forward propagation to satisfy these equations.

2. **Lagrange multipliers** ($\boldsymbol{\lambda}$, excluding $\lambda_n$): Setting the gradient of the $\mathcal{L}$ w.r.t. the intermediate variables equal to zeros tells us what to do with the intermediate multipliers.
   $$
   \begin{eqnarray*}
   0 &=& \nabla_{\! z_j} \mathcal{L} \\
   &=& \nabla_{\! z_j}\! \left[ z_n - \sum_{i=1}^n \lambda_i \cdot \left( z_i - f_i(z_{\alpha(i)}) \right) \right] \\
   (j\ne n)&=& - \sum_{i=1}^n \lambda_i \nabla_{\! z_j}\! \left[ \left( z_i - f_i(z_{\alpha(i)}) \right) \right] \\
   &=& - \left( \sum_{i=1}^n \lambda_i \nabla_{\! z_j}\! \left[ z_i \right] \right) + \left( \sum_{i=1}^n \lambda_i \nabla_{\! z_j}\! \left[ f_i(z_{\alpha(i)}) \right] \right) \\
   &=& - \lambda_j + \sum_{i \in \beta(j)} \lambda_i \frac{\partial f_i(z_{\alpha(i)})}{\partial z_j} \\
   &\Updownarrow& \\
   \lambda_j &=& \sum_{i \in \beta(j)} \lambda_i \frac{\partial f_i(z_{\alpha(i)})}{\partial z_j} \\
   \end{eqnarray*}
   $$
   *Key observation*: The last equation for $\lambda_j$ should look very familiar: It is exactly the equation used in backpropagation! It says that we sum $\lambda_i$ of nodes that immediately depend on $j$ where we scaled each $\lambda_i$ by the derivative of the function that directly relates $i$ and $j$. You should think of the scaling as a "unit conversion" from derivatives of type $i$ to derivatives of type $j$.

3. **Input multipliers** ($λ_{1:d}$): Our dummy constraints gives us $λ_{1:d}$, which are conveniently equal to the gradient of the function we're optimizing:
   $$
   \nabla_{\!\boldsymbol{x}} f(\boldsymbol{x}) = \boldsymbol{\lambda}_{1:d}.
   $$
   

   Of course, this interpretation is only precise when ① the constraints are satisfied (z equations) and ② the linear system on multipliers is satisfied (λ equations).

4. **Input variables** ($\boldsymbol{x}$):  We get these using the $\nabla_{\!\boldsymbol{x}} f(\boldsymbol{x})$'s that we got from the multipliers.

To summarize:

1. Backprop is does not directly fall out of the the rules for differentiation that you learned in calculus (e.g., the chain rule).  This is since it operates on a more general family of functions: programs which have intermediate variables
2. Backprop is a particular instantiation of the method of Lagrange multipliers.

## Variational Inference

#### Variational Inference: Foundation and Modern Methods ([Talk](https://youtu.be/ogdv_6dbvVQ?t=1099), [Slides](<https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf>))

> Inference answers the question: What does this model say about this data.

 The Probabilistic pipe line goes as such:

1. Use knowledge to make assumptions about model (gaussian, iid etc).

2. Use data to train model and discover patterns in the data.

3. Use patterns (or trained model) to predict and explore. (Do Inference!)

4. Criticize model and revise assumptions (1).

   ![Variational inference sketch.pdf](Y:\Code\Markdown\Personal_notes\Deep learning mathematical preliminary\Variational inference sketch.pdf.png)

   

 A probabilistic model is a joint distribution between hidden variables $z$ and observed variables $x$ "what is the probability of having both these $x$ and $z$?"
$$
p(x,z)
$$
Inference about the unknown $z$ is through the *posterior* $p(z|x)$  "given these $x$, what is the probability of $z$?":
$$
p(z|x)=p(x,z)/p(x)
$$
 The biggest problem is that the denominator is not tractable, one cannot compute it, we must thus approximate it, this is **approximate posterior inference**.

> Variational Inference turns Inference in to an Optimization Problem.

We posit a variational  family of distributions over the latent variables:
$$
q(z; \nu)
$$
and **fit** the variational parameter $\nu$ for the variational distribution to be KL close to the exact posterior.  As we see in the figure, start with some family of distributions and approximate them to be as close as space permits to the real posterior.

![Variational optimization](Y:\Code\Markdown\Personal_notes\Deep learning mathematical preliminary\Variational optimization.png)

 An example to variational inference is the Mixture of gaussians which start with a variational family of a few gaussians and approximate their variational parameter $\nu$, which in this case is their mean and STD to fit the data best.

###### Part 1  -Mean Field and Stochastic Variational Inference

 An example we will see along the talk is of *Topic Modeling*, topic models use posterior inference to discover the hidden thematic structure in a large collection of documents. 

The specific model we use is LDA which is based upon the premise that a document has several topics. Each of these topics is some distribution over words , each document is a mixture of topics (distributions), each word is drawn from on of those topics.

![LDA](Y:\Code\Markdown\Personal_notes\Deep learning mathematical preliminary\LDA.png)

When we have the model, we can use it as a posterior distribution, observe a document with all the topics hidden and we want to calculate:
$$
p(\text{topics, proportions, assignments | documents})
$$
Our model is a graphical one, what this means is that it encodes assumptions about the data that helps us to factorize the joint distribution. An example of a graphical model can be seen below: 

![Graphical model](Y:\Markdown\Personal_notes\Deep learning mathematical preliminary\Graphical model.png)

Each node is a random variable, black points are parameters, shaded nodes are observed. The inference process works as such, each topic gets it's parameters from the learned $\eta$, there are $K$ such topics and $\beta$ represents a sample from their distribution ($\beta$ tells us which topic are we working on right now) . 

Each document gets $\theta _d$ from $\alpha$ which is the distribution over topics per-documents, from this distribution, $z$ is sampled and sets the topic assignment, the color, of each word. Combining the per-document dist and the general topic dist eventually allow for the sampling of $w$.

All these parameters and $w$, the word have some joint probability, this graph just makes some order and assumptions about it. The posterior takes the form of:
$$
p(\beta,\theta,z|w)=\frac{p(\beta,\theta,z,w)}{\int_\beta \int_\theta \sum_z p(\beta,\theta,z,w)}
$$
The problem here is that we cannot compute the denominator, the marginal $p(w)$. We will use approximate inference.

How do we get this uncomputable posterior?  

> The goal of inference is to get the model parameters (the hidden variables) given the data $p(\alpha,\beta,\gamma,\delta|x)$.

Since we can't compute the posterior, minimizing the KL div seems to be hard, to bypass this problem, we use the ELBO, evidence lower bound which is a proxy to KL divergence.

> Maximizing the ELBO is  the same as minimizing the KL divergence.

ELBO is of the form:
$$
\mathcal{L}=\mathbb{E}_q[\log p(z,x)] - \mathbb{E}_q[\log q_\theta(z,x)]
$$
First term pushes $q$ to put all it's mass on the MAP estimate of $p$, the second term is actually the entropy of $q$, it pushes it towards a wider distribution.

Optimization of the ELBO can be done using the mean field approximation that sets all latent variables as independent, this allows for iterative coordinate decent optimization.

> In the mean field family, each latent variable has it's own parameter.





















​	

### Normalizing flow

[Notes](<https://deepgenerativemodels.github.io/notes/>)

**In normalizing flow we parametrize the distribution by the transformations from a unit gaussian to it. When we learn, we actually learn the transformations.**

##### Training NF using MLE

Sample $x$ from the data $x\sim p(x)$, transform it to $z$ using $z=f(x)$, then minimize the likelihood of $p_\theta(x)$ given the transformation:
$$
\text{argmax}_\theta \log p_\theta(x) = \text{argmax}_\theta\left[\log p(f_\theta(x))-\log\left|\frac{\part f_\theta(x)}{\part x}\right|\right]
$$
After the optimization over many data points $x$, we have learned $\theta$  and can perform inference (density estimation for data point $x$) using our knowledge of the base distribution (usually a gaussian).

If $f()$ is invertible, we can use it to generate new data by sampling $z$ from the base dist $z_0 \sim p(z_0)$ and transform it according to 
$$
x=f^{-1}_\theta(z)
$$












Why does one needs NF? Since a good estimation of $p(x)$ makes it possible to efficiently complete many downstream tasks: sample unobserved but realistic new data points (data generation), predict the rareness of future events (density estimation), infer latent variables, fill in incomplete data samples, etc.

Machine learning is all about probability, what we do when we train a model is tune the parameters in order to maximize the probability of the training dataset under the model. In order to do so, we must assume some probability distribution as the output of our model. Usually, we choose Categorical distribution for classification and Gaussian for regression (in the gaussian distribution,  taking the (log) maximum likelihood of a gaussian distribution is what gives us the MSE loss). Choosing a gaussian distribution as our distribution model is problematic since a gaussian is a very simple function.

Having established that the Gaussian distribution is sometimes overly simplistic, one wonders if we can find a better distribution which is both complex enough to model rich, multi-modal data distributions like images while retaining the easy comforts of a Normal distribution: sampling, density evaluation, and with re-parameterizable samples? there are a few ways to get by in this path, one of which is by using a mixture model, another is **Normalizing Flows** which enables to learn invertible, volume-tracking transformations of distributions that we can manipulate easily.

The main idea behind Normalizing Flows (NF) is that of change of variables, this is since the transformations between more and more complex distribution must preserve volume, the main tool for Normalizing flow based methods is the change of variables theorem which enables us to move towards more and more elaborate distributions.

![1558354914965](C:\Users\yotampe\AppData\Roaming\Typora\typora-user-images\1558354914965.png)

##### Technically speaking

NF is based upon the Change of variable theorem. The change of variables theorem reduces the whole problem of figuring out the distortion of the content to understanding the infinitesimal distortion, i.e., the distortion of the derivative which is given by the determinant of the Jacobian matrix (a generalization for high dimensionality of the derivative). One should note that this theorem is infinitesimal and does not apply to large transformations.





Normalizing flows are invertible transformations of distributions, they enable one to transform simple probability densities to complex ones through a series of non-linear transformations (in a same manner a NN takes the input and takes it through a set of transformation). 

After such transformation, the distribution satisfy the following formula for variable exchange in distribution:
$$
\ln q_K(z_K)=\ln q_0(z_0)+\sum _{k=1}^K \ln \left| \frac{\part f_k}{\part z_{k-1}}\right |
$$
where $q_0(z_0)$ is the initial distribution and $q_K(z_K)$ is the final transformed distribution.

### Variational autoencoder

Goal, we wish to describe out data using some set of latent variables $z$ and maximize the model $p(X)$ w.r.t $p(x|z)$ and $p(z)$. This poses a big problem since marginalizing over $z$ (doing $p(x)=\sum_z p(x|z)p(z)$ is intractable since there might be a huge number of $z$'s in a multiple dimensional space which summing over is impossible. Solution, defining a separate variational distribution for each $z$, say a gaussian to serve as an approximate posterior $q_i$ , this approximate postirior is used as a lower bound on the real likelihood, it is called the Evidence Lower Bound (ELBO), now ELBO is easier to maximize in small steps where on each step we use a few images to get $z$ and then approximate $q \approx z$ 







##### From a Technical point of view ([A nice blog I followed](http://akosiorek.github.io/ml/2018/03/14/what_is_wrong_with_vaes.html))

###### Latent Variable Models

Say we want to model the world in terms of a probability distribution $p(\mathbf{x})$ with $\mathbf{x}\in R^D$. The thing is - the world is complicated and we do not know what form of $p(\mathbf{x})$ we need take. To account for it, we introduce another variables $\mathbf{z}\in R^d$ which describes, or explains the content of $\mathbf{x}$. For example, in case $\mathbf{x}$ is an image, $\mathbf{x}$ can contain information about the number, type and appearance of objects in the image. In this case, the total probability will be something like, what is the probability of having two cats in the image times the probability of having this image given that there are two cats etc, this makes the distribution more complex, it will take the form:
$$
p(\mathbf{x}) = \int p(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})~d \mathbf{z}.
$$
 This is an example of a mixture model, for every possible value of $\mathbf{z}$ we add another distribution to $p(\mathbf{x})$, weighted by it's probability.  Another use of Latent variables is done when considering an autoencoder, the vector in the middle, between the encoder and the decoder is called the Latent vector since the output distribution will be dependent on them but we do not have them as data points.

The general method to find this $p(x)$ is using maximum likelihood estimation on the parameters of the distribution:
$$
\theta^\star = \arg \max_{\theta \in \Theta} p_\theta(\mathbf{x}).
$$
 The problem with using this practice with the above case is that we cannot evaluate $p_\theta (x)$. In order to evaluate it we can use [importance sampling (IS)](https://en.wikipedia.org/wiki/Importance_sampling). which is a method that allows  to sample from a different probability distribution (*proposal*) and then weigh those samples with respect to the nominal pdf. Let $q_ϕ(z∣x)$ be our proposal - a probability distribution parametrised by a neural network with parameters $ϕ∈Φ$. We can write:
$$
p_\theta(\mathbf{x}) = \int p(\mathbf{z}) p_\theta (\mathbf{x} \mid \mathbf{z})~d \mathbf{z} =\\
  \mathbb{E}_{p(\mathbf{z})} \left[ p_\theta (\mathbf{x} \mid \mathbf{z} )\right] =
  \mathbb{E}_{p(\mathbf{z})} \left[ \frac{q_\phi ( \mathbf{z} \mid \mathbf{x})}{q_\phi ( \mathbf{z} \mid \mathbf{x})} p_\theta (\mathbf{x} \mid \mathbf{z} )\right] =
  \mathbb{E}_{q_\phi ( \mathbf{z} \mid \mathbf{x})} \left[ \frac{p_\theta (\mathbf{x} \mid \mathbf{z} ) p(\mathbf{z})}{q_\phi ( \mathbf{z} \mid \mathbf{x})} )\right].
$$
Above, we see that we actually obtain the marginal probability by sampling over the proposed distribution which we have. 

From [importance sampling literature](http://statweb.stanford.edu/~owen/mc/Ch-var-is.pdf) we know that the optimal proposal $$q^*_ϕ(z∣x)$$ is proportional to the nominal pdf $p_\theta(\mathbf{x})$ times the function, whose expectation we are trying to approximate ( $p_\theta(\mathbf{x} \mid \mathbf{z})$ ).  From Bayes, $p(z \mid x) = \frac{p(x \mid z) p (z)}{p(x)}$, this means that the **optimal proposal is proportional to the posterior distribution which is very hard to find.**

Fortunately, it turns out that by approximating the posterior with a learned proposal, we can approximate the marginal probability $p_\theta (x)$ , this looks like an autoencoding setup.

###### Rise of a Variational Autoencoder

To learn the model, we will need: $p_\theta (x,z)$ - the generative model, which consists of $p_\theta (x|z)$ - a probabilistic decoder and $p(z)$ - a prior over the latent variables. Apart from the above, we will also need $q_\phi	(z|x)$ a probabilistic encoder.  Upon having all these things' we can use the Kulback-Leibler divergence in order to measure the distance between the distributions. 
$$
KL \left( q_\phi (\mathbf{z} \mid \mathbf{x}) || p_\theta(\mathbf{z} \mid \mathbf{x}) \right) = \mathbb{E}_{q_\phi (\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{q_\phi (\mathbf{z} \mid \mathbf{x})}{p_\theta(\mathbf{z} \mid \mathbf{x})} \right]
$$
It seems like we remain with the same problem, we cannot evaluate the posterior distribution. This is where some algebra comes in handy:
$$
\begin{align}
  KL &\left( q_\phi (\mathbf{z} \mid \mathbf{x}) || p_\theta(\mathbf{z} \mid \mathbf{x}) \right)\\
  &=\mathbb{E}_{q_\phi (\mathbf{z} \mid \mathbf{x})} \left[ \log q_\phi (\mathbf{z} \mid \mathbf{x}) - \log p_\theta(\mathbf{z} \mid \mathbf{x}) \right]\\
  &=\mathbb{E}_{q_\phi (\mathbf{z} \mid \mathbf{x})} \left[ \log q_\phi (\mathbf{z} \mid \mathbf{x}) - \log \frac{p_\theta(\mathbf{z}, \mathbf{x})}{ p_\theta(\mathbf{x})} \right]\\
   &=\mathbb{E}_{q_\phi (\mathbf{z} \mid \mathbf{x})} \left[ \log q_\phi (\mathbf{z} \mid \mathbf{x}) - \log p_\theta(\mathbf{z}, \mathbf{x}) \right] + \log p_\theta(\mathbf{x})\\
  &= -\mathcal{L} (\mathbf{x}; \theta, \phi) + \log p_\theta(\mathbf{x})
\end{align}
$$
1->2 line is an expansion of the log, 2->3 is a combination of Bayes theorem and the definition of conditional probability, 3->4 the independency of $p_\theta (x)$ on $z$, and 4->5 the definition of  $\mathcal{L} (\mathbf{x}; \theta, \phi)$ as the "Evidence-lower-bound" (ELBO) which can be now rewritten as:
$$
\log p_\theta(\mathbf{x}) = \mathcal{L} (\mathbf{x}; \theta, \phi) + KL \left( q_\phi (\mathbf{z} \mid \mathbf{x}) || p_\theta(\mathbf{z} \mid \mathbf{x}) \right)\\
$$
and 
$$
\mathcal{L} (\mathbf{x}; \theta, \phi) = \mathbb{E}_{q_\phi (\mathbf{z} \mid \mathbf{x})}
    \left[
      \log \frac{
        p_\theta (\mathbf{x}, \mathbf{z})
      }{
        q_\phi (\mathbf{z} \mid \mathbf{x})
      }
    \right]
$$
Upon approximating it using just one sample from the proposal distribution, we get:
$$
\mathcal{L} (\mathbf{x}; \theta, \phi) \approx  \log \frac{
      p_\theta (\mathbf{x}, \mathbf{z})
    }{
      q_\phi (\mathbf{z} \mid \mathbf{x})
    }, \qquad \mathbf{z} \sim q_\phi (\mathbf{z} \mid \mathbf{x})
$$
ELBO is used as a proxy which we can evaluate:
$$
\phi^\star,~\theta^\star = \arg \max_{\phi \in \Phi,~\theta \in \Theta}
  \mathcal{L} (\mathbf{x}; \theta, \phi)
$$
By maximizing ELBO, we:

1. Maximize the marginal probability (which is what we actually came to do!)
2. Minimize the KL-divergence, or both.

The nice thing here is that once the distribution of latent variables is indeed similar to the proor distribution we choose for $z$ ,once we want to generate an example we just generate it using $p(z)$

##### From a intuitive point of view

The difference between a Variational autoencoder and a regular autoencoder is that is the former we put a constraint on the encoding network which forces it to generate latent vectors that roughly follow a unit gaussian distribution, a continues latent space. This makes creating images rather easy, we can just sample a vector from a unit gaussian and know that we will not get trash out of the decoder. Note that STD=1 is large and the network will usually want to have it -> 0.  This comes to solve the fundamental problem with autoencoders, for generation, is that the latent space they convert their inputs to and where their encoded vectors lie, may not be continuous, or allow easy interpolation.

![1557639034295](C:\Users\yotampe\AppData\Roaming\Typora\typora-user-images\1557639034295.png)

 Generally, as one might expect, there will be a tradeoff between how well does the latent variables match a unit gaussian and how accurate is the network (this is since we do not let the net to create the best latent vectors). 

In fact, by adding another term to the loss that forces the unit gaussian distribution, we let the network make this balance (according to the $\lambda$ we chose), the loss will take the following form: 

```python
generation_loss = mean(square(generated_image - real_image))  
latent_loss = KL-Divergence(latent_variable, unit_gaussian)  
loss = generation_loss + latent_loss  
```

In order to simplify the KL divergence calculations, instead of having a single latent vector, for each image, the encoder will output a mean vector and a standard deviation vector from which the sample latent vector is sampled, this is different from the previous  case in that there is a non zero STD, and the latent vector is sampled instead of set.

```python
# z_mean and z_stddev are two vectors generated by encoder network
latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)  
```

When we're calculating loss for the decoder network, we can just sample from the standard deviations and add the mean, and use that as our latent vector:

```python
# The samples for the decoder network are taken from the distributions by sampling from a unit gaussian and adding the mean and std the encoder gave.
samples = tf.random_normal([batchsize,n_z],0,1,dtype=tf.float32) 
sampled_z = z_mean + (z_stddev * samples)  
```

In addition to allowing us to generate random latent variables, this constraint also improves the generalization of our network.

### RNN Encoder-Decoder

Based on the [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](<https://arxiv.org/pdf/1406.1078.pdf>) in which the authors propose a novel neural network architecture that learns to *encode* a variable-length sequence into a fixed-length vector representation and  to decode a  given  fixed-length  vector  representation back into a variable-length sequence.

From a probabilistic point of view, this model enables one to learn a conditional distribution over a variable length sequence conditioned on yet another variable length sequence:
$$
p(y_1,...,y_{T'}\ |\ x_1,...,x_T)
$$
where one should note that $T\ne T'$. In order to achieve the above objective, two RNNs are trained, the first, the *encoder* is a simple RNN that is trained to read the the data sequentially, update the hidden state according to the new data point and the last activation and eventually output a *summary vector* $\boldsymbol{c}$. the decoder is another RNN which is trained to generate the output sequence by predicting the next symbol $y_t$ given the hidden state  $h_{\left< t \right>}$ **and** $y_{t-1}$ **and** $\boldsymbol{c}$ as opposed to a standard RNN that only gets the previous state and incoming part of sequence.  This construction intuitively means that each node knows now about both the current state of the sentence (up to it's time point) but also about the reminder of the sentence, the hidden layer's activation in this case takes the form of:
$$
h_{\left< t \right >}=f(h_{\left< t-1 \right >},y_{t-1},\boldsymbol{c}).
$$
and the conditional probability of the next symbol is:
$$
p(y_t|y_t-1,...,y_1,\boldsymbol{c})=g(h_{\left<t-1\right>},y_{t-1},\boldsymbol{c})
$$
![1557651199749](C:\Users\yotampe\AppData\Roaming\Typora\typora-user-images\1557651199749.png)

The encoder-decoder are jointly trained to maximize the conditional log-likelihood:
$$
\max _\theta \frac{1}{N}\sum_{n=1}^N \log p_\theta(\boldsymbol{y}_n|\boldsymbol{x}_m)
$$
Ones the model is trained, it can be used in two ways:

1. Generate a target sequence given an input one (translate)
2. Score a given pair of sequences using the above $p_\theta(\boldsymbol{y}|\boldsymbol{x})$ .

## Recurrent Neural Networks

from [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](<https://arxiv.org/pdf/1406.1078.pdf>):

> A recurrent neural network (RNN) is a neural network that consists of a hidden state $h$ and and optional output $y$ which operates on a variable length sequence $\boldsymbol{x}=(x_1,...,x_T)$. At each time step $t$, the hidden state $h_{\left<t\right>}$ of the RNN is updated by 
> $$
> h_{\left< t \right >}=f(h_{\left< t-1 \right >},x_t)
> $$
> where $f$ can go from a simple nonlinear activation function such a tanh or Relu to a complicated long sort term memory unit (LSTM).
>
> An RNN can learn a probability distribution over a sequence by being trained to predict the next symbol in a sequence, the distribution one learns in this case is $p(x_t|x_t-1,...,x_1)$. Using the conditional probability,  we can compute the probability of a sequence $\boldsymbol{x}$ by:
> $$
> p(\boldsymbol{x})=\Pi _{t=1}^{T}p(x_t|x_t-1,...,x_1)
> $$
> From this learned distribution, it is straightforward to sample a new sequence by iteratively sampling a symbol at each time step. 

## Generative Adversarial Networks

## Convolutional neural networks

###  Transposed convolution

Transposed convolution / Deconvolution / Fractionally strided convolution- a learnable way to up sample ([Really great video from Stanford](https://youtu.be/nDPWywWRIRo?t=1370)) 

Instead of stride preforming down sampling as in a regular convolution, in transposed convolution, the stride is an up sampling factor. In transposed convolution, we take the smaller input, multiply each cell with the weights and add the results to the output. Note that we sum over the overlaps.

![1557220443504](C:\Users\yotampe\AppData\Roaming\Typora\typora-user-images\1557220443504.png)

 Why Transposed? it turns out that when the matrix operation which is the convolution is transposed, it's operation is the transposed convolution.

![1557220665288](C:\Users\yotampe\AppData\Roaming\Typora\typora-user-images\1557220665288.png)





## Neural ordinary differential equations

#### ODE Talk by Duvanov

Intuition - it is easier to learn the change to improve an almost correct answer (expanding near unity ) then it is to compleatly transform the answer at every level. Also, it makes it easier for the gradients to flow this way.

The first step in turning a resnet into a ODENet is to reparametrize the layers s.t they will also know at which time they are at, instead of inputting $(z,\theta[t])$ we input $([z,t],\theta)$, this is simply reparametrization but it actually means that the state has time tied to it and the layer gets time as an input. After this substitution, the process looks like an Euler solver and can be replaced by and ODE solver. What is nice with the modern methods for solving ODEs is that they are dynamic in the sense that given an input, they can change the number of evaluations (layers). 

The authors are not the first to think about using an continuous time model instead of a discreet one, their contibution comes from the way to train this model and baypass the need to backpropagate through the layers. This is hard from two reasons: 1.  High memory cost in order to keep all the intermediate values 2. the solver add extra numerical error for every step but eventually converges to the right answer. A better way is to approximate the derivative and should not derive the approximations.

The way that is proposed to learn using this method is using the adjoint sensitivity method which is a continuous take on the regular back propagation, in this method there are extra two differential equations to solve, one for the adjoint sensitivity ($\delta$ in the regular backprop) and another one, using this adjoint for $\frac{\part L}{\part \theta}$. 

Another benifite here is constant memory cost, this is since, once the forward pass is done, all the backwards pass needs to know is the last hidden state, the ODE solver can get the states as the reverse pass goes along by solving the original ODE backwards in time.

ODENets does not have a notion of depth, the thing that is closes to depth is evaluations, another nice thing is that in the process of learning, the network makes half the backwards pass than forward passes and saves time by doing it.

#### Continuous time dynamics

Unlike ResNets, ODENets have a well defined state at all times, even between observations.  This enables them to extrapolate data better then an RNN that doesn't work well with irregular time interval. One can even take advantage of the data that comes with the irregularity and add a Poisson process, the usual problem with adding a Poisson likelihood integral is that it has a time integral but now this fits really nicely.

**How does the RNN takes care of the irregular intervals**

##### Density modeling 

 Usually, in order to preform a change of variables, one has to take the determinent of the Jacobian of the operation, this is a rather expansive feat. The authors found out that in the case that the change in $z$ is instatanouos, the change of variables becomes much easier to compute



### Implicit Vs Explicit ODE solvers

Given a differential equation $\frac{\part u}{\part t}=f(u,t)$, constraining $f(u,t)=-c u(t)$ where   $c$ is positive, gives the equation:
$$
\frac{\part u(t)}{\part t}=-cu(t)
$$
 This differential equation might be solved using an **explicit scheme** in which the next step is evaluated using only the value of the past states:
$$
u_{n+1}=u_n-c\Delta t u_n=u_n(1-c\Delta t)
$$
A the von Neumann stability method for ODEs uses the growth factor, here, it is:
$$
\lambda=\frac{u_{n+1}}{u_n}=1-c\Delta t
$$
 the scheme is stable for $|\lambda|<1$ neutral when $|\lambda|=1$ and unstable when $|\lambda|>1$, we see that in this case, the scheme may be unstable (notice the  absolute value on the $\lambda$).

An **Explicit scheme** will be of the sort
$$
u_{n+1}=u_n-c\Delta t u_{n+1}\Rightarrow u_{n+1}=\frac{u_n}{1+c\Delta t}
$$
in this case the growth factor is $\lambda=\frac{1}{1+c\Delta t}$ which always has a absolute value greater than 1.

Thus, it seems that an implicit method is better at solving this equation, it is however, not without falls since is will add more computational complexity since division is harder that multiplication computationally. (Note, it seems to me that regarding the Euler methods here, the explicit method is just the 2nd order approximation of the implicit one). Implicit methods is used when larger steps are wanted, each step will be harder to compute but there my be less steps so this could become worthwhile.

**The difference between the two methods is whether we use the past value ($u_n$) or the next value ($u_{n+1}$) in order to evaluate the next ($u_{n+1}$)**



# Toward Theoretical Understanding of Deep Learning (ICML 2018 tutorial)

##### Optimization of DNNs

There are a  lot of direction in $d$ dimensions:

in $R^d$ , $\exist \ \exp(\frac{d}{\epsilon})$ directions whose pairwise angle is at most $\epsilon$ degrees. This makes the general notion of arriving at an answer naively intractable.

There is no knowledge of the loss landscape, the optimization algorithm works in a black box.

To ensure so  decent in GD we put a bound on the Hessian with doesn't allow the landscape to fluctuate a lot. The talk has a small proof that shows that if the hessian is bound by some $\beta$ there is a set difference on the loss between two steps which means that so long as the gradient is large you will make some progress. Note that this doesn't mean arriving at a global minimum, however, this does not pose a problem since it has been shown that upon addition of some noise (as we naturally have), the optimization passes all saddle points and reaches a bottom of some valley. 

2nd order optimization methods also exist, however, they do not give better answers to SGD is still used.

It has been shown that in single layer nets SGD is promised to arrive at a global minimum.

##### Role of depth

The ideal result in this field is to find some problem which cannot be solved with depth $d$ and can be solved with a net of depth $d+1$.

Pros of depth is more expressivity, Cons are more difficult optimization. A recent paper showed that increasing network depth can accelerate optimization.

##### Theory Generative models and GANs

Unsupervised learning is sometimes called Representation learning and lies on the notion of Manifold assumption that the data has some underlying structure. The goal is to use a large **unlabeled** dataset and learn this manifold, which is the same as learning a code mapping from the data to some latent variables in a latent space.

The hope is that the latent variables, or code is a good substitution for the data. Say if the data has only black dogs and white cats, it is enough to have a color variable instead of the full image, the code is much thinner but still holds all the data.

Deep generative models try to solve this code, their first assumption is that the distribution of codes (latent variables) is a random vector (sampled from some distribution) with mapping done by a DNN.

