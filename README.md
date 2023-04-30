## Guided music score-performance translation model.

An encoder-decoder transformer model attempting to learn to generate musical performances making use of the Museformer attention scheme to improve accuracy.
We pre-train the decoder block of our model on the large [Giant-Midi dataset](https://github.com/bytedance/GiantMIDI-Piano) to improve the generated pieces quality while avoiding overfitting. And then make use of the [ASAP dataset](https://github.com/fosfrancesco/asap-dataset) to learn a translation between the midi-score or a piece and the midi recordings from different performances of the piece.

We make use of the Octuple tokenisation method, and use the [Museformer](https://github.com/microsoft/muzic/tree/main/museformer) attention scheme to reduce memory requirements and improve the quality of the pieces we generate.

### Setup

#### Data
The ASAP dataset can be acquired by running the ```download.sh``` file seen in this dataset. For pretraining however, you will need to download the [Giant-Midi Dataset](https://github.com/bytedance/GiantMIDI-Piano/blob/master/disclaimer.md) and create a metadata file containing the list of file paths by:
```cmd
find * > test.csv
sed -i -e '1imidi_filename' test.csv
```

Any other dataset can be setup to work with this project, changes to the data paths can be made in the configs file.


#### Requirements
Install the required packages from the requirements file and ensure you have a CUDA compatible environment set-up.

```cmd
pip install -r requirements.txt
```


### Usage

The training loop of the model can be run by the ```./train/train.py``` with the configs file as argument:
```cmd
python ./train/train.py ./configs/asap.yaml
```
Inside the configs file, you can change the hyperparameters of the model, the dataset file locations, checkpointing variables, and set the project to pre-train or train the full architecture. If ```pretrain``` is ```True``` only the decoder block will be trained and the model will be trained to self-predict the input. Whereas if pretrain ```pretrain``` is ```False``` we will train against the ASAP dataset, with source tokens being the score-midi's and targets the performance-midis.
