# stylegan2-tf-2.x
* This is tensorflow 2.0 based keras subclassing reimplementation of official [StyleGAN2 Repo](https://github.com/NVlabs/stylegan2)

## Inference from official weights
* Please check [how_to_extract_official_weights.md](./how_to_extract_official_weights.md)

| Official repo result | Official weight copied result from this repo |
| :---: | :---: |
| ![official_result] | ![restored_result] |

## Training on 256x256
* batch size 32
* around 300k steps

| d_loss | g_loss |
| :---: | :---: |
| ![256x256_d_loss]| ![256x256_g_loss]|

* result samples

| | | |
| :---: | :---: | :---: |
| ![256x256_result_0]| ![256x256_result_1]| ![256x256_result_2]|

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
[256x256_d_loss]: assets/d_loss_256x256.png
[256x256_g_loss]: assets/g_loss_256x256.png
[256x256_result_0]: assets/out_256x256_0.png
[256x256_result_1]: assets/out_256x256_1.png
[256x256_result_2]: assets/out_256x256_2.png