# stylegan2-tf-2.x
* This is tensorflow 2.0 based keras subclassing reimplementation of official [StyleGAN2 Repo](https://github.com/NVlabs/stylegan2)

## Inference from official weights
| Official repo result | Official weight copied result from this repo |
| :---: | :---: |
| ![official_result] | ![restored_result] |

## current version of training not yet tested!!

## Etc
### Using with pycharm
* If .bashrc file's paths don't work in pycharm environment (e.g. `nvcc -h`), open pycharm with following
```bash
# move to pycharm installed location (location may vary)
moono@moono-ubuntu:~$ cd .local/share/JetBrains/Toolbox/apps/PyCharm-P/ch-0/201.7846.77/bin/

# launch pycharm with .bashrc 
moono@moono-ubuntu:~/.local/share/JetBrains/Toolbox/apps/PyCharm-P/ch-0/201.7846.77/bin$ bash pycharm.sh
```

[loss_tensorboard]: assets/tf-keras-stylegan2-loss.PNG
[generation_tensorboard]: assets/tf-keras-stylegan2-fake-images.PNG
[official_result]: assets/seed6600-official.png
[restored_result]: assets/seed6600-restored.png