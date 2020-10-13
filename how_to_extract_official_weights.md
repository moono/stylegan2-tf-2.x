# Steps to extract official weights

### Prerequisite
* install docker >= 19.03
* install nvidia-docker native

## Step 1. Working on official repository
* Clone official repo
```bash
~$ git clone https://github.com/NVlabs/stylegan2.git
~$ cd stylegan2
```
* Download official weights (*.pkl)
* Create directory
```bash
~$ mkdir official_pretrained
```
* Place official weight (*.pkl) to `<PATH-TO-OFFICIAL-REPO>/official_pretrained`
* Create python script called `export_weights.py` and add following
```python
import tensorflow as tf
import pretrained_networks
import pprint


network_pkl = './official_pretrained/stylegan2-ffhq-config-f.pkl'
print('Loading networks from "%s"...' % network_pkl)
_G, _D, Gs = pretrained_networks.load_networks(network_pkl)
print()

# Print network details.
print('===_G===')
_G.print_layers()
print('===Gs===')
Gs.print_layers()

t_var = tf.trainable_variables()
pprint.pprint(t_var)

saver = tf.train.Saver()

save_path = saver.save(tf.get_default_session(), "./model.ckpt")
```
* Create docker environment for stylegan2
```bash
~$ docker build -f Dockerfile -t stylegan2:official .
``` 
* Get inside docker container and run script
```bash
~$ docker run --gpus all --rm -u $(id -u):$(id -g) -it -v <PATH-TO-OFFICIAL-REPO>:/work-dir -w /work-dir stylegan2:official /bin/bash

# now you are inside docker container
# run following python script to extract model weights
~$ python export_weights.py
``` 
* Check that 4 files are created
  * `checkpoint`
  * `model.ckpt.data-00000-of-00001`
  * `model.ckpt.index`
  * `model.ckpt.meta`
* Leave docker container
```bash
~$ exit
```

## Step 2. Working on this repository(stylegan2-tf-2.x)
* Move to `stylegan2-tf-2.x` directory
```bash
~$ cd <PATH-TO-stylegan2-tf-2.x>
```
* Copy 4 files that are created from step 1 to `<PATH-TO-stylegan2-tf-2.x>/official-pretrained`
* Create docker environment for stylegan2-tf-2.x
```bash
~$ docker build -f Dockerfile -t stylegan2:tf.2.x .
```
* Get inside docker container and run script
```bash
~$ docker run --gpus all --rm -u $(id -u):$(id -g) -it -v <PATH-TO-stylegan2-tf-2.x>:/work-dir -w /work-dir stylegan2:tf.2.x /bin/bash

# now you are inside docker container
# run following python script to convert model weights
~$ python inference_from_official_weights.py
``` 
* Check inference result images `*.png`
* Check converted model weights `./official-converted/cuda` && `./official-converted/ref` 