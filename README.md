# stylegan2-tf-2.x
* StyleGAN2 in tf-2.0 keras subclassing

## Implemented
* Generator
  * conv weight modulation / demodulation
  * skip architecture
* Discriminator
  * resnet architecture

## Not implemented yet
* Path regularization
* Lazy regularization
* Multi GPU distributed training
* Fast optimized bias / activation / bilinear filtering (no cuda code)
  * Using official code's slow reference code
