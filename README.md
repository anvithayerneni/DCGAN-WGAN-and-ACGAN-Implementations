# DCGAN, WGAN, and ACGAN Implementations
This repository contains the implementation of Deep Convolutional Generative Adversarial Networks (DCGAN), Wasserstein Generative Adversarial Networks (WGAN), and Auxiliary Classifier Generative Adversarial Networks (ACGAN) for generating synthetic images as part of the deep learning coursework.

## Dependencies
Ensure you have the following Python libraries installed:

torch
torchvision
numpy
matplotlib

## Models Overview
DCGAN Configuration
Epochs: 15
Learning Rate: 0.0002
Generator: Utilizes transposed convolution layers to upscale a noise vector to a 64x64 image.
Discriminator: Consists of convolution layers that downscale the input image, assessing its authenticity.
WGAN Configuration
Epochs: 15
Learning Rate: 0.0002
Noise Dimension: 128
Weight Clipping Limit: 0.01
Generator and Discriminator: Structured similarly to DCGAN with modifications to stabilize training.
ACGAN Configuration
Epochs: 20
Generator: Features multiple layers of transposed convolution to transform a labeled noise vector into a 64x64 image, leveraging batch normalization and ReLU activations.
Discriminator: Comprises convolutional layers followed by a dual pathway for authenticity check and label prediction using sigmoid and softmax activations.

## Performance
Inception Score: Both DCGAN and WGAN models achieved an Inception Score of 0.3678794503211975.
Frechet Inception Distance (FID): Computationally intensive, representing the similarity between real and generated image sets.
