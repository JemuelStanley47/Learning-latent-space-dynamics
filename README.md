# Learning-latent-space-dynamics
Implementation of a Variational Autoencoder for learning latent space dynamics from images to push

<img src="img/overview.png" width=600>

### Data collection
The actions that are uniformly random sampled within the action space limits and the corresponding state are collected
<img src="img/cam_sample.png" width=600>

### Action-Space
Each action 
```
$\mathbf u = \begin{bmatrix} p & \phi & \ell\end{bmatrix}^\top\in \mathbb R^3$ is composed by:
* $p \in [-1,1]$: pushing location along the lower block edge.
* $\phi \in [-\frac{\pi}{2},\frac{\pi}{2}]$ pushing angle.
* $\ell\in [0,1]$ pushing length as a fraction of the maximum pushing length. The maximum pushing length is is 0.1 m
```
<img src="img/state-action-space.png" width=300 alt="Action space">

## VAE
The VAE Encoder, which maps images to a Gaussian distribution over latent vectors is implemented. The encoder outputs $\mu$ and $\log\sigma^2$ which parameterize the latent distribution.The architecture is shown below:
<img src="img/VAE_Arch.png" width=600>

### State Decoder
<img src="img/state_decoder_arch.png" width=600>

### State Encoder
<img src="img/state_encoder_arch.png" width=600>

## RESULTS
### Reconstructed image
<img src="img/reconstructed_img.png" width=600>

### Image-based controller



