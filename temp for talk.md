$$
\mathbf{z}_t=\mathbf{z}_{t-1}+f(\mathbf{z}_{t-1},\theta,t)
$$

$$
\mathbf{z}(t+h)=\mathbf{z}(t)+hf(\mathbf{z},t)
$$

$$
f(\mathbf{z}_t,\theta)
$$

$$
f(\mathbf{z}_t,\theta[t])
$$


$$
\mathbf{z}_t-\mathbf{z}_{t-1}=f(\mathbf{z}_{t-1},\theta,t)
$$

$$
f(\mathbf{z}_{t-1},\theta,t)=\mathbf{z}_t-\mathbf{z}_{t-1}=\Delta \mathbf{z}_t
$$

$$
\mathbf{z}_t-\mathbf{z}_{t-1}=f(\mathbf{z}_{t-1},\theta,t)
$$

$$
f(\mathbf{z}_{t-1},\theta,t)=\mathbf{z}_t
$$



$$
\frac{d\mathbf{z}_t}{dt}=f(\mathbf{z}_{t},\theta,t)
$$

$$
\mathbf{z}(t_N)=\mathbf{z}(t_0)+\int_{t_0}^{t_N} dt'\  f(\mathbf{z}(t'),\theta,t')
$$

$$
\frac{\part L}{\part z_t}=\frac{\part L}{\part z_{t+1}}\frac{\part z_{t+1}}{\part z_t}
$$

$$
\mathbf{z}(t)=\mathbf{z}(t_N)+\int_{t_N}^{t} dt'\  f(\mathbf{z}(t'),\theta,t')
$$


$$
\frac{\part L}{\part \theta_t}=\frac{\part L}{\part z_{t}}\frac{\part f(z_{t},\theta_t)}{\part \theta_t}
$$

$$
Z=f_\theta(X)  \\ X'=g_\phi(Z)
$$

$$
\frac{dx(t)}{dt}=v(t)
$$

$$
\begin{align}
x(t)&=x_{t_0}+\int_{t_0}^t dt'\ v(t')\\
&=x_{t_0}+\lim_{\Delta t\rightarrow 0} \Delta t\sum_{t'=t_0}^t \ v(t')
\end{align}
$$

$$
=x_{t_0}+\lim_{\Delta t\rightarrow 0} \Delta t\sum_{t'=t_0}^t \ v(t')
$$

$$
\frac{dx}{dt}=v(t)\ \ \Leftrightarrow\ \ \frac{d\mathbf{z}_t}{dt}=f(\mathbf{z}_{t},\theta,t)
$$

$$
\text{Error}\propto h^2
$$

$$
a(t_N)=\frac{dL}{d\mathbf{z}(t_N)}
$$

$$
a(t)=a(t_N)-\int_{t_N}^{t}dt'a(t')^T\frac{\part f(\mathbf{z}(t'),t',\theta) }{\part \mathbf z(t')}
$$

$$
\frac{dL}{d\theta}=-\int_{t_N}^{t_0}dt'a(t')^T\frac{\part f(\mathbf{z}(t'),t',\theta) }{\part \theta}
$$

$$
\mu\ \ \ \ \sigma
$$

$$
\begin{align}
\mathcal {L}(\mathbf {\phi } ,\mathbf {\theta } ,\mathbf {x} )=
&-\mathbb {E} _{q_{\phi }(\mathbf {z} |\mathbf {x} )}{\big (}\log p_{\theta }(\mathbf {x} |\mathbf {z} ) \big)
\\&\ +D_{\mathrm {KL} }(q_{\phi }(\mathbf {z} |\mathbf {x} )\Vert p_{\theta }(\mathbf {z} ))
\end{align}
$$

$$
{\displaystyle {\mathcal {L}}(\mathbf {\phi } ,\mathbf {\theta } ,\mathbf {x} )=-\mathbb {E} _{q_{\phi }(\mathbf {z} |\mathbf {x} )}{\big (}\log p_{\theta }(\mathbf {x} |\mathbf {z} ){\big )}}
$$

$$
\tilde{L}\sim4L
$$

$$
\mathbf{z}_t=(z^1,z^2,z^3,z^4,z^5,z^6)^T
\\
[\mathbf{z}_t,t]={(z^1,z^2,z^3,z^4,z^5,z^6,t)}^T
$$

