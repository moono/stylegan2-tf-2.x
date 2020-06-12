# stylegan2-tf-2.x
* This is tensorflow 2.0 based keras subclassing reimplementation of official [StyleGAN2 Repo](https://github.com/NVlabs/stylegan2)

## To do list
| Items | Implemented |
| :--- |  :---: |
| weight modulation / demodulation | :heavy_check_mark: |
| skip architecture | :heavy_check_mark: |
| resnet architecture | :heavy_check_mark: |
| Path regularization | :heavy_check_mark: |
| Lazy regularization | :heavy_check_mark: |
| Fast optimized bias / activation (cuda compiled code) | - |
| Fast optimized bilinear filtering (cuda compiled code) | :heavy_check_mark: |
| Single GPU training | :heavy_check_mark: |
| Multi GPU distributed training | :heavy_check_mark: |
| Inference from official Generator weights | :heavy_check_mark: |

## Inference from official weights
| Official repo result | Official weight copied result from this repo |
| :---: | :---: |
| ![official_result] | ![restored_result] |

## Previous implementation result (current version of training not tested!!)
| at 284k train step | Screenshot |
| :--- |  :---: |
| **Loss** |  ![loss_tensorboard] |
| **Generation output**<br><br>Real Images<br><br>phi=0.0<br><br>phi=0.5<br><br>phi=0.7<br><br>phi=1.0  | ![generation_tensorboard] |

[loss_tensorboard]: assets/tf-keras-stylegan2-loss.PNG
[generation_tensorboard]: assets/tf-keras-stylegan2-fake-images.PNG
[official_result]: assets/seed6600-official.png
[restored_result]: assets/seed6600-restored.png