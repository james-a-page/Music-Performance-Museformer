# Music Performance
Maps musical scores to performances, conditioned on instructions.

## Dataset Creation
There is currently only support for the [ASAP dataset](https://github.com/fosfrancesco/asap-dataset), which can be downloaded in the respective directory. Other datasets will be added soon.

## Training
Change the config file as desired, then run:

`python3 train/train.py configs/asap.yaml`

Train, validation loss, and greedy decoding are reported during training using `tensorboard`.